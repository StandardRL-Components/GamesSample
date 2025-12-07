import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:50:12.667686
# Source Brief: brief_01242.md
# Brief Index: 1242
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import copy

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Activate nodes in the correct sequence to establish a connection before the timer runs out. "
        "Rewind time to correct your mistakes."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to activate a node and shift to rewind time."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 8
    GRID_ROWS = 5
    NODE_RADIUS = 15
    CURSOR_SPEED = 0.2  # Interpolation speed
    MAX_STEPS = 1000
    FPS = 30 # Assumed FPS for visual timing

    # --- Colors ---
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (30, 50, 80)
    COLOR_CONNECTION = (50, 80, 120)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_GLOW = (255, 255, 0, 60)

    NODE_COLORS = {
        "inactive": (200, 50, 50),   # Red
        "active": (50, 200, 50),     # Green
        "transmitting": (50, 150, 255), # Blue
        "error": (255, 150, 0),      # Yellow
    }
    
    # --- State Enums ---
    STATE_INACTIVE = 0
    STATE_ACTIVE = 1
    STATE_TRANSMITTING = 2
    STATE_ERROR = 3
    STATE_MAP = {
        STATE_INACTIVE: "inactive",
        STATE_ACTIVE: "active",
        STATE_TRANSMITTING: "transmitting",
        STATE_ERROR: "error",
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.successful_transmissions = 0
        
        self.nodes = []
        self.connections = []
        self.target_sequence = []
        self.activated_indices = []
        
        self.cursor_grid_pos = [0, 0]
        self.cursor_screen_pos = [0.0, 0.0]

        self.particles = []
        self.history = []
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reward_this_step = 0
        self.timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.particles.clear()
        self.history.clear()
        self.activated_indices.clear()

        self._generate_puzzle()
        
        self.cursor_grid_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.cursor_screen_pos = list(self._get_screen_pos_for_grid(self.cursor_grid_pos))

        self._add_history_snapshot()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        self._handle_input(movement, space_press, shift_press, action)

        # --- Update Game Logic ---
        self._update_game_state()
        self._add_history_snapshot()

        # --- Termination and Rewards ---
        self.timer -= 1
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated and not self.game_over:
            if self.timer <= 0:
                self.reward_this_step += -100  # Timeout failure
                # sfx: game_over_failure
            elif len(self.activated_indices) == len(self.target_sequence):
                self.reward_this_step += 100  # Success
                self.successful_transmissions += 1
                # sfx: game_over_success
            self.game_over = True
        
        self.score += self.reward_this_step

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _generate_puzzle(self):
        self.nodes.clear()
        self.connections.clear()
        
        # Create nodes
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                screen_pos = self._get_screen_pos_for_grid([c, r])
                self.nodes.append({"grid_pos": [c, r], "screen_pos": screen_pos, "state": self.STATE_INACTIVE, "anim_timer": 0})

        # Create connections (ensure a connected graph)
        for i, node in enumerate(self.nodes):
            c, r = node["grid_pos"]
            if c < self.GRID_COLS - 1: # Connect right
                neighbor_idx = i + 1
                self.connections.append((i, neighbor_idx))
            if r < self.GRID_ROWS - 1: # Connect down
                neighbor_idx = i + self.GRID_COLS
                self.connections.append((i, neighbor_idx))

        # Generate target sequence (solvable path)
        num_activations = 3 + (self.successful_transmissions // 2)
        num_activations = min(num_activations, self.GRID_COLS * self.GRID_ROWS)
        
        path = []
        start_node_idx = self.np_random.integers(0, len(self.nodes))
        current_idx = start_node_idx
        
        while len(path) < num_activations:
            if current_idx not in path:
                path.append(current_idx)
            
            possible_neighbors = [n_idx for c_idx, n_idx in self.connections if c_idx == current_idx] + \
                                 [c_idx for c_idx, n_idx in self.connections if n_idx == current_idx]
            
            if not possible_neighbors: break # Should not happen in this grid
            
            current_idx = self.np_random.choice(possible_neighbors)

        self.target_sequence = path
        
        # Set timer based on difficulty
        base_time = 30 * self.FPS # 30 seconds
        time_reduction = (5 * self.FPS) * (self.successful_transmissions // 5) # 5s reduction
        self.timer = max(10 * self.FPS, base_time - time_reduction) # 10s minimum

    def _handle_input(self, movement, space_press, shift_press, action):
        # --- Cursor Movement ---
        if movement == 1 and self.cursor_grid_pos[1] > 0: self.cursor_grid_pos[1] -= 1 # Up
        elif movement == 2 and self.cursor_grid_pos[1] < self.GRID_ROWS - 1: self.cursor_grid_pos[1] += 1 # Down
        elif movement == 3 and self.cursor_grid_pos[0] > 0: self.cursor_grid_pos[0] -= 1 # Left
        elif movement == 4 and self.cursor_grid_pos[0] < self.GRID_COLS - 1: self.cursor_grid_pos[0] += 1 # Right
        
        # --- Actions ---
        if space_press:
            # sfx: activate_attempt
            self._activate_node_at_cursor()
        
        if shift_press:
            # sfx: rewind_time
            self._rewind_time()

        self.prev_space_held = action[1] == 1
        self.prev_shift_held = action[2] == 1

    def _activate_node_at_cursor(self):
        node_idx = self.cursor_grid_pos[1] * self.GRID_COLS + self.cursor_grid_pos[0]
        node = self.nodes[node_idx]

        if node_idx in self.activated_indices:
            return # Cannot activate an already active node

        # Check if this is the correct next node in the sequence
        next_target_idx = len(self.activated_indices)
        if next_target_idx < len(self.target_sequence) and node_idx == self.target_sequence[next_target_idx]:
            # Correct activation
            node["state"] = self.STATE_ACTIVE
            node["anim_timer"] = 1.0 # Start transmit animation
            self.activated_indices.append(node_idx)
            self.reward_this_step += 1.0  # Correct node
            self.reward_this_step += 5.0  # Key node bonus
            self._spawn_particles(node["screen_pos"], self.NODE_COLORS["active"], 20)
            # sfx: activate_success
        else:
            # Incorrect activation
            node["state"] = self.STATE_ERROR
            node["anim_timer"] = 1.0 # Start error animation
            self.reward_this_step -= 0.1
            self._spawn_particles(node["screen_pos"], self.NODE_COLORS["error"], 10, speed=2)
            # sfx: activate_fail

    def _rewind_time(self):
        if len(self.history) > 1:
            self.history.pop() # Remove current state
            last_state = self.history[-1] # Get previous state
            
            # Restore state
            self.score = last_state["score"]
            self.timer = last_state["timer"]
            self.activated_indices = last_state["activated_indices"]
            self.cursor_grid_pos = last_state["cursor_grid_pos"]
            for i, node_state in enumerate(last_state["nodes_state"]):
                self.nodes[i]["state"] = node_state
                self.nodes[i]["anim_timer"] = 0
            
            # Visual feedback for rewind
            self._spawn_particles(self.cursor_screen_pos, (200, 200, 255), 30, speed=-3)

    def _update_game_state(self):
        # Interpolate cursor
        target_pos = self._get_screen_pos_for_grid(self.cursor_grid_pos)
        self.cursor_screen_pos[0] += (target_pos[0] - self.cursor_screen_pos[0]) * self.CURSOR_SPEED
        self.cursor_screen_pos[1] += (target_pos[1] - self.cursor_screen_pos[1]) * self.CURSOR_SPEED

        # Update node animations
        for node in self.nodes:
            if node["anim_timer"] > 0:
                node["anim_timer"] -= 1.0 / (0.5 * self.FPS) # 0.5 second animation
                if node["anim_timer"] <= 0:
                    node["anim_timer"] = 0
                    if node["state"] == self.STATE_ERROR:
                        node["state"] = self.STATE_INACTIVE # Revert error state
                    elif node["state"] == self.STATE_ACTIVE:
                        node["state"] = self.STATE_ACTIVE # Stays active

        # Update particles
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _check_termination(self):
        success = len(self.activated_indices) == len(self.target_sequence)
        timeout = self.timer <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        return success or timeout or max_steps_reached
    
    def _add_history_snapshot(self):
        if len(self.history) > self.FPS * 10: # Limit history to 10 seconds
            self.history.pop(0)
        
        snapshot = {
            "score": self.score,
            "timer": self.timer,
            "cursor_grid_pos": list(self.cursor_grid_pos),
            "activated_indices": list(self.activated_indices),
            "nodes_state": [n["state"] for n in self.nodes]
        }
        self.history.append(snapshot)
    
    def _get_observation(self):
        # --- Main Rendering Call ---
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render connections
        for start_idx, end_idx in self.connections:
            start_node = self.nodes[start_idx]
            end_node = self.nodes[end_idx]
            
            color = self.COLOR_CONNECTION
            if start_node["state"] == self.STATE_ACTIVE and end_node["state"] == self.STATE_ACTIVE:
                color = self.NODE_COLORS["active"]

            pygame.draw.aaline(self.screen, color, start_node["screen_pos"], end_node["screen_pos"])

        # Render nodes
        for node in self.nodes:
            pos = (int(node["screen_pos"][0]), int(node["screen_pos"][1]))
            state_str = self.STATE_MAP[node["state"]]
            color = self.NODE_COLORS[state_str]
            
            radius = self.NODE_RADIUS
            if node["anim_timer"] > 0:
                # Pulse effect
                pulse = math.sin(node["anim_timer"] * math.pi)
                radius = int(self.NODE_RADIUS * (1 + 0.3 * pulse))
                if node["state"] == self.STATE_ACTIVE:
                    # Transmitting effect
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(self.NODE_RADIUS * (1 + (1-node["anim_timer"]) * 1.5)), self.NODE_COLORS["transmitting"])

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))
            
        # Render cursor
        pos = (int(self.cursor_screen_pos[0]), int(self.cursor_screen_pos[1]))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.NODE_RADIUS + 8, self.COLOR_CURSOR_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.NODE_RADIUS + 5, self.COLOR_CURSOR)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.NODE_RADIUS + 6, self.COLOR_CURSOR)

    def _render_ui(self):
        # --- Top Bar ---
        bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, (0,0,0,150), bar_rect)
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_sec = self.timer / self.FPS
        timer_color = (255, 255, 255) if time_sec > 5 else (255, 100, 100)
        timer_text = self.font_ui.render(f"TIME: {time_sec:.1f}s", True, timer_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))
        
        # --- Bottom Bar: Transmission Info ---
        bottom_bar_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, (0,0,0,150), bottom_bar_rect)
        
        title_text = self.font_ui.render("TARGET SEQUENCE:", True, (200, 200, 200))
        self.screen.blit(title_text, (10, self.SCREEN_HEIGHT - 30))

        # Draw transmission sequence
        for i, node_idx in enumerate(self.target_sequence):
            x_pos = title_text.get_width() + 25 + i * 30
            y_pos = self.SCREEN_HEIGHT - 25
            
            is_active = i < len(self.activated_indices)
            color = self.NODE_COLORS["active"] if is_active else self.NODE_COLORS["inactive"]
            
            pygame.gfxdraw.filled_circle(self.screen, x_pos, y_pos, 10, color)
            pygame.gfxdraw.aacircle(self.screen, x_pos, y_pos, 10, color)

        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            success = len(self.activated_indices) == len(self.target_sequence)
            msg = "TRANSMISSION COMPLETE" if success else "CONNECTION LOST"
            color = self.NODE_COLORS["active"] if success else self.NODE_COLORS["error"]
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(text_surf, text_rect)
            self.screen.blit(overlay, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "activated_nodes": len(self.activated_indices),
            "target_nodes": len(self.target_sequence),
            "successful_transmissions": self.successful_transmissions
        }

    def _get_screen_pos_for_grid(self, grid_pos):
        c, r = grid_pos
        margin_x = 80
        margin_y = 70
        w = self.SCREEN_WIDTH - 2 * margin_x
        h = self.SCREEN_HEIGHT - 2 * margin_y - 40 # Account for UI bars
        
        x = margin_x + c * (w / (self.GRID_COLS - 1)) if self.GRID_COLS > 1 else margin_x + w / 2
        y = margin_y + r * (h / (self.GRID_ROWS - 1)) if self.GRID_ROWS > 1 else margin_y + h / 2
        return (x, y)

    def _spawn_particles(self, pos, color, count, speed=5, life=15, radius=3):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(0.5, 1.0) * speed
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * vel_mag, math.sin(angle) * vel_mag],
                "life": life,
                "max_life": life,
                "color": color,
                "radius": radius
            })
    
    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("System Grid")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Remove the validation call from the main execution block
    # env.validate_implementation() 
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()