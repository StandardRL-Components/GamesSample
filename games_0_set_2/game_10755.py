import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:00:51.739198
# Source Brief: brief_00755.md
# Brief Index: 755
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Node:
    """Represents a junction, source, or sink in the pipe network."""
    def __init__(self, x, y, node_type='junction'):
        self.pos = np.array([x, y], dtype=float)
        self.type = node_type  # 'source', 'junction', 'sink'
        self.direction = random.randint(0, 3)  # 0:Up, 1:Right, 2:Down, 3:Left
        self.is_active = False

class Pipe:
    """Represents a connection between two nodes."""
    def __init__(self, start_node_idx, end_node_idx, nodes):
        self.start_idx = start_node_idx
        self.end_idx = end_node_idx
        self.start_pos = nodes[start_node_idx].pos
        self.end_pos = nodes[end_node_idx].pos
        self.flow = 0.0
        
        self.vector = self.end_pos - self.start_pos
        self.length = np.linalg.norm(self.vector)
        if self.length > 0:
            self.normal = self.vector / self.length
        else:
            self.normal = np.array([0, 0])

class Particle:
    """Represents a unit of fluid flowing through a pipe."""
    def __init__(self, pos, vel, life, color):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.life = life
        self.max_life = life
        self.color = color

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Rotate pipe junctions to create a path from the source to the sink. "
        "Achieve the target pressure before time runs out."
    )
    user_guide = (
        "Controls: Use ←→ or ↑↓ arrow keys to select a junction. "
        "Press space to rotate clockwise and shift to rotate counter-clockwise."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_PIPE = (50, 60, 80)
        self.COLOR_PIPE_OUTLINE = (30, 40, 60)
        self.COLOR_NODE = (100, 110, 130)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SELECTOR = (255, 255, 0)
        self.COLOR_TARGET = (255, 200, 0)
        self.COLOR_SUCCESS = (0, 255, 128)
        self.COLOR_FAIL = (255, 50, 50)
        
        # Game state that persists across episodes
        self.level = 1
        self.total_score = 0
        
        # Initialize state variables
        self.nodes = []
        self.pipes = []
        self.particles = []
        self.adj = {}
        self.source_node_idx = 0
        self.sink_node_idx = 0
        self.selected_node_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.current_pressure = 0.0
        self.last_pressure = 0.0
        self.target_pressure = 50.0
        self.max_steps = 1800 # 30 seconds at 60fps
        
        # self.reset() is called by the environment wrapper, no need to call it here
        # self.validate_implementation() is for debugging, not needed in final version
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = None # 'success' or 'fail'
        
        self._generate_level()
        
        self.particles = []
        self.selected_node_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.current_pressure = 0.0
        self.last_pressure = 0.0
        
        self._update_flow_and_pressure()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If game is over, an action should reset the level
            if self.win_state == 'success':
                self.level += 1
            obs, info = self.reset()
            return obs, 0, True, False, info

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle input and get immediate rewards
        reward = self._handle_input(movement, space_held, shift_held)
        
        # Update game logic
        self.steps += 1
        self._update_particles()
        
        # Calculate continuous rewards
        reward += self._calculate_reward()
        
        # Check for termination
        terminated = self._check_termination()
        if terminated:
            reward += 100 if self.win_state == 'success' else -100

        self.score += reward
        self.last_pressure = self.current_pressure
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.nodes.clear()
        self.pipes.clear()
        self.adj.clear()

        num_nodes = min(15, 2 + self.level)
        self.target_pressure = min(200, 40 + self.level * 10)

        # Create nodes
        if self.level < 5: # Linear layout
            for i in range(num_nodes):
                x = self.WIDTH * (i + 1) / (num_nodes + 1)
                y = self.HEIGHT / 2
                self.nodes.append(Node(x, y))
        else: # Branching layout
            node_positions = []
            while len(node_positions) < num_nodes:
                pos = (self.np_random.uniform(50, self.WIDTH - 50), self.np_random.uniform(50, self.HEIGHT - 50))
                if all(np.linalg.norm(np.array(pos) - np.array(p)) > 75 for p in node_positions):
                    node_positions.append(pos)
            
            node_positions.sort(key=lambda p: p[0]) # Sort by x-pos to ensure clear source/sink
            for pos in node_positions:
                self.nodes.append(Node(pos[0], pos[1]))

        self.source_node_idx = 0
        self.sink_node_idx = num_nodes - 1
        self.nodes[self.source_node_idx].type = 'source'
        self.nodes[self.sink_node_idx].type = 'sink'
        self.nodes[self.sink_node_idx].direction = -1 # Sink has no direction

        # Create pipes
        if self.level < 5:
            for i in range(num_nodes - 1):
                self.pipes.append(Pipe(i, i + 1, self.nodes))
        else: # Guaranteed connected graph (DAG)
            for i in range(num_nodes - 1):
                j = self.np_random.integers(i + 1, num_nodes)
                self.pipes.append(Pipe(i, j, self.nodes))
            # Ensure sink is reachable
            if not any(p.end_idx == self.sink_node_idx for p in self.pipes):
                 i = self.np_random.integers(0, num_nodes - 1)
                 self.pipes.append(Pipe(i, self.sink_node_idx, self.nodes))


        # Create adjacency list for fast lookup
        for i in range(len(self.nodes)):
            self.adj[i] = []
        for i, pipe in enumerate(self.pipes):
            self.adj[pipe.start_idx].append(i)

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        action_taken = False

        # --- Node Selection ---
        junction_indices = [i for i, n in enumerate(self.nodes) if n.type == 'junction']
        if not junction_indices:
            return 0 # No junctions to interact with

        current_selection_idx = -1
        if self.selected_node_idx in junction_indices:
            current_selection_idx = junction_indices.index(self.selected_node_idx)
        else:
            # If current selection is not a junction, snap to the first one
            current_selection_idx = 0
            self.selected_node_idx = junction_indices[0]

        if movement in [1, 4]: # Up or Right -> Next
            current_selection_idx = (current_selection_idx + 1) % len(junction_indices)
            self.selected_node_idx = junction_indices[current_selection_idx]
        elif movement in [2, 3]: # Down or Left -> Previous
            current_selection_idx = (current_selection_idx - 1 + len(junction_indices)) % len(junction_indices)
            self.selected_node_idx = junction_indices[current_selection_idx]

        # --- Arrow Rotation ---
        node = self.nodes[self.selected_node_idx]
        if node.type == 'junction':
            old_pressure = self.current_pressure
            # On press (0 -> 1 transition)
            if space_held and not self.last_space_held:
                node.direction = (node.direction + 1) % 4
                action_taken = True
            if shift_held and not self.last_shift_held:
                node.direction = (node.direction - 1 + 4) % 4
                action_taken = True

            if action_taken:
                self._update_flow_and_pressure()
                if self.current_pressure > old_pressure:
                    reward += 1.0 # Event-based reward for a good move
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _update_flow_and_pressure(self):
        for pipe in self.pipes:
            pipe.flow = 0.0
        for node in self.nodes:
            node.is_active = False

        q = [(self.source_node_idx, 1.0)] # (node_idx, flow_strength)
        visited = set()

        while q:
            current_idx, flow_in = q.pop(0)
            if current_idx in visited: continue
            visited.add(current_idx)

            node = self.nodes[current_idx]
            node.is_active = True

            if node.type == 'sink':
                continue

            # Determine which outbound pipe matches the node's direction
            vec_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)} # Up, Right, Down, Left
            arrow_vec = np.array(vec_map[node.direction])

            best_pipe_idx = -1
            max_dot = -2 # dot product range is [-1, 1]

            for pipe_idx in self.adj.get(current_idx, []):
                pipe = self.pipes[pipe_idx]
                dot = np.dot(arrow_vec, pipe.normal)
                if dot > max_dot:
                    max_dot = dot
                    best_pipe_idx = pipe_idx
            
            if best_pipe_idx != -1 and max_dot > 0.5: # Must be a decent match
                pipe = self.pipes[best_pipe_idx]
                pipe.flow = flow_in
                q.append((pipe.end_idx, flow_in))

        # Calculate total pressure at the sink
        total_pressure = 0
        for pipe in self.pipes:
            if pipe.end_idx == self.sink_node_idx:
                total_pressure += pipe.flow
        self.current_pressure = total_pressure * 50 # Scale pressure for display

    def _update_particles(self):
        # Spawn new particles
        for pipe in self.pipes:
            if pipe.flow > 0 and self.np_random.random() < 0.5:
                num_particles = int(pipe.flow * 2)
                for _ in range(num_particles):
                    start_pos = pipe.start_pos + pipe.normal * self.np_random.uniform(0, 10)
                    vel = pipe.normal * (2 + pipe.flow * 3)
                    life = pipe.length / np.linalg.norm(vel) if np.linalg.norm(vel) > 0 else 100
                    color = self._get_flow_color(pipe.flow)
                    self.particles.append(Particle(start_pos, vel, life, color))

        # Move and cull existing particles
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.pos += p.vel
            p.life -= 1
    
    def _calculate_reward(self):
        reward = 0
        if self.current_pressure > self.last_pressure:
            reward += 0.1
        elif self.current_pressure < self.last_pressure:
            reward -= 0.1
        return reward
    
    def _check_termination(self):
        if self.game_over:
            return True
        if self.current_pressure >= self.target_pressure:
            self.game_over = True
            self.win_state = 'success'
        elif self.steps >= self.max_steps:
            self.game_over = True
            self.win_state = 'fail'
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "pressure": self.current_pressure,
            "target_pressure": self.target_pressure
        }

    def _render_game(self):
        # Render pipes
        for pipe in self.pipes:
            color = self._get_flow_color(pipe.flow)
            pygame.draw.line(self.screen, self.COLOR_PIPE_OUTLINE, pipe.start_pos, pipe.end_pos, 16)
            pygame.draw.line(self.screen, color if pipe.flow > 0 else self.COLOR_PIPE, pipe.start_pos, pipe.end_pos, 10)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (p.color[0], p.color[1], p.color[2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), 3, color)

        # Render nodes
        for i, node in enumerate(self.nodes):
            color = self.COLOR_SUCCESS if node.is_active else self.COLOR_NODE
            pygame.gfxdraw.filled_circle(self.screen, int(node.pos[0]), int(node.pos[1]), 15, color)
            pygame.gfxdraw.aacircle(self.screen, int(node.pos[0]), int(node.pos[1]), 15, self.COLOR_PIPE_OUTLINE)
            
            # Render arrows for junctions
            if node.type == 'junction':
                self._render_arrow(node.pos, node.direction, 10, self.COLOR_TEXT)
            elif node.type == 'source':
                pygame.draw.rect(self.screen, self.COLOR_TEXT, (node.pos[0]-5, node.pos[1]-5, 10, 10))


        # Render selector
        junction_indices = [i for i, n in enumerate(self.nodes) if n.type == 'junction']
        if not self.game_over and self.selected_node_idx in junction_indices:
            node_pos = self.nodes[self.selected_node_idx].pos
            pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
            radius = int(20 + pulse * 5)
            alpha = int(100 + pulse * 155)
            pygame.gfxdraw.aacircle(self.screen, int(node_pos[0]), int(node_pos[1]), radius, (*self.COLOR_SELECTOR, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(node_pos[0]), int(node_pos[1]), radius-1, (*self.COLOR_SELECTOR, alpha))

    def _render_ui(self):
        # Level and Score
        level_text = self.font_medium.render(f"LEVEL {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (20, 10))
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 10))

        # Timer
        time_left = (self.max_steps - self.steps) / 60.0
        timer_color = self.COLOR_FAIL if time_left < 10 else self.COLOR_TEXT
        timer_text = self.font_medium.render(f"TIME: {time_left:.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH / 2 - timer_text.get_width()/2, 10))

        # Pressure Gauge
        gauge_pos = (self.WIDTH / 2, self.HEIGHT - 40)
        gauge_radius = 60
        max_pressure_display = self.target_pressure * 1.2
        
        # Gauge background
        pygame.draw.arc(self.screen, self.COLOR_PIPE, (*(gauge_pos - np.array([gauge_radius, gauge_radius])), gauge_radius*2, gauge_radius*2), math.radians(150), math.radians(390), 10)
        
        # Current pressure arc
        pressure_angle = 150 + (self.current_pressure / max(1, max_pressure_display)) * 240
        pressure_angle = min(pressure_angle, 390)
        if pressure_angle > 150:
            pygame.draw.arc(self.screen, self.COLOR_SUCCESS, (*(gauge_pos - np.array([gauge_radius, gauge_radius])), gauge_radius*2, gauge_radius*2), math.radians(150), math.radians(pressure_angle), 10)
        
        # Target pressure marker
        target_angle = math.radians(150 + (self.target_pressure / max(1, max_pressure_display)) * 240)
        target_pos_outer = gauge_pos + np.array([math.cos(target_angle), -math.sin(target_angle)]) * (gauge_radius + 8)
        target_pos_inner = gauge_pos + np.array([math.cos(target_angle), -math.sin(target_angle)]) * (gauge_radius - 8)
        pygame.draw.line(self.screen, self.COLOR_TARGET, target_pos_inner, target_pos_outer, 3)

        # Pressure text
        pressure_text = self.font_medium.render(f"{self.current_pressure:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(pressure_text, (gauge_pos[0] - pressure_text.get_width()/2, gauge_pos[1] - 15))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.win_state == 'success':
            text = self.font_large.render("SUCCESS!", True, self.COLOR_SUCCESS)
            self.total_score += self.score
        else:
            text = self.font_large.render("TIME UP", True, self.COLOR_FAIL)
        
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
        self.screen.blit(text, text_rect)

        prompt_text = self.font_small.render("Press any action to continue", True, self.COLOR_TEXT)
        prompt_rect = prompt_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
        self.screen.blit(prompt_text, prompt_rect)

    def _get_flow_color(self, flow):
        if flow <= 0: return self.COLOR_PIPE
        flow = min(1.0, flow)
        # Interpolate Blue -> Green -> Red
        if flow < 0.5:
            # Blue to Green
            r = int(0 + 70 * (flow * 2))
            g = int(100 + 155 * (flow * 2))
            b = int(200 - 100 * (flow * 2))
        else:
            # Green to Red
            r = int(70 + 185 * ((flow - 0.5) * 2))
            g = int(255 - 205 * ((flow - 0.5) * 2))
            b = int(100 - 100 * ((flow - 0.5) * 2))
        return (r, g, b)

    def _render_arrow(self, pos, direction, size, color):
        points = []
        if direction == 0: # Up
            points = [(pos[0], pos[1] - size), (pos[0] - size/2, pos[1]), (pos[0] + size/2, pos[1])]
        elif direction == 1: # Right
            points = [(pos[0] + size, pos[1]), (pos[0], pos[1] - size/2), (pos[0], pos[1] + size/2)]
        elif direction == 2: # Down
            points = [(pos[0], pos[1] + size), (pos[0] - size/2, pos[1]), (pos[0] + size/2, pos[1])]
        elif direction == 3: # Left
            points = [(pos[0] - size, pos[1]), (pos[0], pos[1] - size/2), (pos[0], pos[1] + size/2)]
        
        if points:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not run in a headless environment.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pipe Flow")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Player Input ---
        movement = 0 # no-op
        space_held = False
        shift_held = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = True
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = True

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            pygame.time.wait(2000) # Pause for 2 seconds on win/loss
        
        clock.tick(60) # Run at 60 FPS
        
    pygame.quit()