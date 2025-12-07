import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:22:21.578439
# Source Brief: brief_02401.md
# Brief Index: 2401
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Coordinate three workers to synergistically collect scattered resources.

    This is a turn-based strategy/puzzle game on a 7x7 grid. The agent controls
    three workers, one at a time, to collect 10 resources within a 30-turn limit.

    Key Mechanics:
    - Each full turn consists of one action for each of the three workers.
    - Workers have a budget of 2 "move points" per turn.
    - A standard move (up, down, left, right) costs 1 move point.
    - An "assist" action (pushing another worker) costs 2 move points.
    - The game ends in victory if 10 resources are collected.
    - The game ends in failure if the 30-turn limit is reached or if any
      two workers become separated by more than 4 units (Manhattan distance).

    Action Space (`MultiDiscrete([5, 2, 2])`):
    - `action[0]` (Movement):
        - 0: No-op (for Move or Assist)
        - 1-4: Direction (Up, Down, Left, Right)
    - `action[1]` (Space):
        - 1: "Pass" - the active worker forfeits all remaining move points for the turn.
    - `action[2]` (Shift):
        - 1: "Assist" - attempts to push the next worker in the specified direction.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Coordinate three workers to collect scattered resources on a grid. "
        "Manage their movement and use teamwork to gather all resources before time runs out or they become too separated."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press 'space' to pass a worker's turn. "
        "Hold 'shift' and press an arrow key to perform an assist action, pushing another worker."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 7
    CELL_SIZE = 50
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (50, 50, 70)
    COLOR_RESOURCE = (255, 215, 0)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_PANEL = (40, 40, 60, 180)
    WORKER_COLORS = [(255, 90, 90), (90, 255, 90), (90, 150, 255)]
    COLOR_ACTIVE_GLOW = (255, 255, 255)
    COLOR_ASSIST_EFFECT = (180, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.render_mode = render_mode

        # Game state variables are initialized in reset()
        self.workers = []
        self.worker_moves_left = []
        self.resources = set()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turn_number = 0
        self.active_worker_idx = 0
        self.particles = []
        self.visual_effects = []
        
        self.max_turns = 30
        self.resources_to_win = 10
        self.max_separation = 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turn_number = 1
        self.active_worker_idx = 0
        self.particles = []
        self.visual_effects = []
        
        # Initialize workers
        start_positions = [(3, 2), (3, 3), (3, 4)]
        self.workers = [{'pos': list(p), 'color': c} for p, c in zip(start_positions, self.WORKER_COLORS)]
        self.worker_moves_left = [2, 2, 2]
        
        # Initialize resources
        self.resources = set()
        occupied_cells = set(start_positions)
        while len(self.resources) < self.resources_to_win:
            pos = (self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE))
            if pos not in occupied_cells:
                self.resources.add(pos)
                occupied_cells.add(pos)

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.visual_effects.clear()

        active_worker = self.workers[self.active_worker_idx]
        moves_left = self.worker_moves_left[self.active_worker_idx]

        action_taken = False
        if moves_left > 0:
            if space_held: # Pass action
                self.worker_moves_left[self.active_worker_idx] = 0
                action_taken = True
            elif shift_held and movement != 0: # Assist action
                if moves_left >= 2:
                    target_idx = (self.active_worker_idx + 1) % 3
                    target_worker = self.workers[target_idx]
                    
                    dx, dy = self._get_move_delta(movement)
                    new_pos = [target_worker['pos'][0] + dx, target_worker['pos'][1] + dy]

                    if self._is_valid_pos(new_pos):
                        target_worker['pos'] = new_pos
                        self.worker_moves_left[self.active_worker_idx] -= 2
                        # Add visual effect for assist
                        start_px = self._grid_to_pixel(active_worker['pos'])
                        end_px = self._grid_to_pixel(new_pos)
                        self.visual_effects.append({'type': 'assist', 'start': start_px, 'end': end_px, 'life': 5})
                        # Check for resource collection by assisted worker
                        if tuple(new_pos) in self.resources:
                            self._collect_resource(tuple(new_pos))
                            reward += 1.0
                        action_taken = True
                    else:
                        reward -= 0.01 # Penalty for invalid assist
                else:
                    reward -= 0.01 # Penalty for trying to assist with insufficient moves
            elif movement != 0: # Move action
                if moves_left >= 1:
                    dx, dy = self._get_move_delta(movement)
                    new_pos = [active_worker['pos'][0] + dx, active_worker['pos'][1] + dy]
                    
                    if self._is_valid_pos(new_pos):
                        active_worker['pos'] = new_pos
                        self.worker_moves_left[self.active_worker_idx] -= 1
                        if tuple(new_pos) in self.resources:
                            self._collect_resource(tuple(new_pos))
                            reward += 1.0
                        action_taken = True
                    else:
                        reward -= 0.01 # Penalty for invalid move

        # Advance to next worker's action
        self.active_worker_idx = (self.active_worker_idx + 1) % 3
        self.steps += 1

        # Check for end of a full turn (after worker 2 has acted)
        if self.active_worker_idx == 0:
            self.turn_number += 1
            self.worker_moves_left = [2, 2, 2] # Reset move points for all workers

        # Check for termination conditions
        if self.score >= self.resources_to_win:
            self.game_over = True
            reward += 100.0 # Victory bonus
        elif self.turn_number > self.max_turns:
            self.game_over = True
            reward -= 100.0 # Loss penalty
        elif self._check_worker_separation():
            self.game_over = True
            reward -= 100.0 # Separation penalty
        
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._update_and_draw_particles()
        self._render_resources()
        self._render_workers()
        self._render_visual_effects()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "turn": self.turn_number}

    # --- Helper and Rendering Methods ---

    def _grid_to_pixel(self, grid_pos):
        x = self.GRID_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _get_move_delta(self, movement_action):
        if movement_action == 1: return 0, -1  # Up
        if movement_action == 2: return 0, 1   # Down
        if movement_action == 3: return -1, 0  # Left
        if movement_action == 4: return 1, 0   # Right
        return 0, 0

    def _is_valid_pos(self, pos):
        return 0 <= pos[0] < self.GRID_SIZE and 0 <= pos[1] < self.GRID_SIZE

    def _check_worker_separation(self):
        pos = [w['pos'] for w in self.workers]
        dist1 = abs(pos[0][0] - pos[1][0]) + abs(pos[0][1] - pos[1][1])
        dist2 = abs(pos[0][0] - pos[2][0]) + abs(pos[0][1] - pos[2][1])
        dist3 = abs(pos[1][0] - pos[2][0]) + abs(pos[1][1] - pos[2][1])
        return max(dist1, dist2, dist3) > self.max_separation

    def _collect_resource(self, pos):
        if pos in self.resources:
            self.resources.remove(pos)
            self.score += 1
            # Spawn particles on collection
            px_pos = self._grid_to_pixel(pos)
            for _ in range(20):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(15, 30)
                self.particles.append({'pos': list(px_pos), 'vel': vel, 'life': life, 'color': self.COLOR_RESOURCE})
            # sfx: resource_collect.wav

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, self.GRID_Y_OFFSET), (start_x, self.GRID_Y_OFFSET + self.GRID_HEIGHT), 1)
            # Horizontal lines
            start_y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, start_y), (self.GRID_X_OFFSET + self.GRID_WIDTH, start_y), 1)

    def _render_resources(self):
        radius = self.CELL_SIZE // 4
        for res_pos in self.resources:
            px, py = self._grid_to_pixel(res_pos)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_RESOURCE)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_RESOURCE)

    def _render_workers(self):
        size = self.CELL_SIZE * 0.7
        radius = int(size * 0.25)
        for i, worker in enumerate(self.workers):
            px, py = self._grid_to_pixel(worker['pos'])
            rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            
            # Active worker glow
            if i == self.active_worker_idx and not self.game_over:
                glow_radius = int(size * 0.6 + 5 * math.sin(self.steps * 0.2))
                alpha = int(100 + 50 * math.sin(self.steps * 0.2))
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, self.COLOR_ACTIVE_GLOW + (alpha,), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Worker body
            pygame.draw.rect(self.screen, worker['color'], rect, border_radius=radius)
            # Worker moves indicator
            if not self.game_over:
                moves = self.worker_moves_left[i]
                for m in range(moves):
                    dot_x = rect.left + (m + 1) * (size / 3)
                    dot_y = rect.top - 8
                    pygame.draw.circle(self.screen, worker['color'], (dot_x, dot_y), 3)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                color = p['color'] + (alpha,)
                try:
                    pygame.gfxdraw.pixel(self.screen, int(p['pos'][0]), int(p['pos'][1]), color)
                except TypeError: # Color might not have alpha
                    pygame.gfxdraw.pixel(self.screen, int(p['pos'][0]), int(p['pos'][1]), p['color'])

    def _render_visual_effects(self):
        for fx in self.visual_effects[:]:
            fx['life'] -= 1
            if fx['life'] <= 0:
                self.visual_effects.remove(fx)
            elif fx['type'] == 'assist':
                alpha = int(255 * (fx['life'] / 5))
                pygame.draw.line(self.screen, self.COLOR_ASSIST_EFFECT + (alpha,), fx['start'], fx['end'], 4)
                # sfx: assist_zap.wav

    def _render_ui(self):
        # UI Panel
        panel_surf = pygame.Surface((self.SCREEN_WIDTH, self.GRID_Y_OFFSET), pygame.SRCALPHA)
        panel_surf.fill(self.COLOR_UI_PANEL)
        self.screen.blit(panel_surf, (0, 0))

        # Turn counter
        turn_text = f"TURN: {self.turn_number} / {self.max_turns}"
        self._draw_text(turn_text, (self.SCREEN_WIDTH // 2, 25), self.font_large, self.COLOR_UI_TEXT)

        # Score
        score_text = f"RESOURCES: {self.score} / {self.resources_to_win}"
        self._draw_text(score_text, (self.SCREEN_WIDTH // 2, 55), self.font_small, self.COLOR_UI_TEXT)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.score >= self.resources_to_win else "GAME OVER"
            self._draw_text(msg, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2), pygame.font.Font(None, 72), self.COLOR_RESOURCE if self.score >= self.resources_to_win else self.WORKER_COLORS[0])
            
    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block is for manual play and will not be run by the evaluation system.
    # It requires a display.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Play Controls ---
    # Arrow Keys: Move/Set Assist Direction
    # Space: Pass current worker's turn
    # Left Shift: Perform Assist action
    # Q: Quit
    
    # Mapping from Pygame keys to action components
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Use a real display for manual play
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Worker Coordination Game")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    # Remove the validation call from the main execution block
    # env.validate_implementation() 
    
    while not done:
        movement_action = 0
        space_action = 0
        shift_action = 0

        # Event handling for manual control
        event_processed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_movement:
                    movement_action = key_to_movement[event.key]
                    event_processed = True
                elif event.key == pygame.K_SPACE:
                    space_action = 1
                    event_processed = True
                elif event.key == pygame.K_q:
                    done = True
                    break
        
        if done:
            break

        # Get pressed keys for continuous actions (like holding shift)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
        
        # If a key was pressed, take a step
        if event_processed:
            action = [movement_action, space_action, shift_action]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Turn: {info['turn']}, Score: {info['score']}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            done = terminated or truncated
        
        # Render the environment to the real screen
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit to 15 FPS for turn-based game

    if 'info' in locals():
        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    env.close()