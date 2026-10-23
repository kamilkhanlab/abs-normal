using LinearAlgebra, XLSX, PlotlyJS, CSV, DataFrames

function plot_Kuramoto(
    time_data, 
    raw_data, 
    solver_data
)
    p = make_subplots(rows = 2, cols = 1, subplot_titles = ["Raw Data Coupled Trajectory" "ReLU NN Coupled Trajectory"])
    num_lines = size(raw_data, 2) 
    
    base_colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "gray", "olive"]
    color_palette = [base_colors[mod1(i, length(base_colors))] for i in 1:50]

    for i in 1:num_lines
        add_trace!(p, 
            scatter(x = time_data, y = raw_data[:, i], mode = "lines", showlegend = false, line_color = color_palette[i]), 
            row = 1, col = 1
        )
    end
    for i in 1:num_lines
        add_trace!(p, 
            scatter(x = time_data, y = solver_data[:, i], mode = "lines", showlegend = false, line_color = color_palette[i]), 
            row = 2, col = 1
        )
    end

    relayout!(p, 
        font = attr(
            family = "Computer Modern, Latin Modern Roman, Times New Roman, serif",
            size = 14
        ),
        
        xaxis = attr(showline=true, mirror=true, linecolor="black", gridcolor="#f0f0f0", title="Time", showgrid=false),
        yaxis = attr(showline=true, mirror=true, linecolor="black", gridcolor="#f0f0f0", title="sin(θ)", showgrid=false),
        xaxis2 = attr(showline=true, mirror=true, linecolor="black", gridcolor="#f0f0f0", title="Time", showgrid=false),
        yaxis2 = attr(showline=true, mirror=true, linecolor="black", gridcolor="#f0f0f0", title="sin(θ)", showgrid=false),

        plot_bgcolor = "white"
    )

    return p
end #function 

Z = 61; dt = 0.0005714; occurrences=70_000;
time_length = collect(0:dt:(occurrences*dt))

df_output = CSV.read("Kuromoto_output.csv", DataFrame, header=false)
df_raw = CSV.read(string("data/DataSet", Z,"/FitRNet_Storage_RawData.csv"), CSV.Tables.matrix; header=false);

p1 = plot_Kuramoto(time_length, sin.(df_raw), sin.(df_output))
