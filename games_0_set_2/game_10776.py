import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:55:39.269890
# Source Brief: brief_00776.md
# Brief Index: 776
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    ChromaShift: A puzzle game where the player manipulates rotating mirrors
    to guide a color-changing light beam. The goal is to match a target
    color by collecting light from various energy sources, all within a
    limited number of turns.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Mirror rotation (0=no-op, 1-4=rotate mirror 1-4)
    - actions[1]: Unused (Space button)
    - actions[2]: Unused (Shift button)

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where the player manipulates rotating mirrors to guide a color-changing "
        "light beam to match a target color."
    )
    user_guide = (
        "Press keys 1-4 to rotate the corresponding mirrors. Each rotation uses one turn. "
        "Try to match the target color with your beam."
    )
    auto_advance = False


    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_CELL_SIZE = 80
    MAX_TURNS = 20
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (15, 18, 23)
    COLOR_GRID = (40, 45, 55)
    COLOR_MIRROR = (200, 200, 220)
    COLOR_MIRROR_ACTIVE = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    BEAM_INITIAL_COLOR = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Game state variables are initialized in reset()
        self.mirrors = []
        self.energy_sources = []
        self.target_color = (0, 0, 0)
        self.turns_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.beam_path = []
        self.beam_color = (0, 0, 0)
        self.last_rotated_mirror = -1
        self.last_rotated_timer = 0
        self.win_state = False

        # Fixed positions for puzzle elements
        self.GRID_W = self.WIDTH // self.GRID_CELL_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_CELL_SIZE
        self.BEAM_ORIGIN_CELL = (0, self.GRID_H // 2)
        self.BEAM_INITIAL_DIR = (1, 0)  # Right

        self._mirror_positions = [(2, 1), (2, 3), (4, 2), (6, 1), (6, 3)]
        self._source_colors = [
            (100, 0, 0), (0, 100, 0), (0, 0, 100), (50, 50, 0), (50, 0, 50),
            (0, 50, 50), (150, 0, 0), (0, 150, 0), (0, 0, 150), (75, 75, 75)
        ]
        self._source_positions = [
            (1, 0), (1, 4), (3, 0), (3, 4), (5, 0), (5, 4),
            (7, 0), (7, 2), (7, 4), (3, 2)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.turns_left = self.MAX_TURNS
        self.game_over = False
        self.win_state = False
        self.last_rotated_mirror = -1
        self.last_rotated_timer = 0

        # Initialize mirrors (5 mirrors, 4 controllable)
        self.mirrors = []
        for i, pos in enumerate(self._mirror_positions):
            self.mirrors.append({
                "pos": pos,
                "angle": self.np_random.integers(0, 4), # 0,1,2,3 for 0,90,180,270 deg
                "controllable": i < 4
            })

        # Initialize energy sources
        self.energy_sources = []
        for i, pos in enumerate(self._source_positions):
            self.energy_sources.append({
                "pos": pos,
                "color": self._source_colors[i],
                "collected": False
            })

        # Generate an achievable target color
        num_sources_for_target = self.np_random.integers(3, 6)
        target_indices = self.np_random.choice(len(self.energy_sources), num_sources_for_target, replace=False)
        self.target_color = [0, 0, 0]
        for i in target_indices:
            self.target_color[0] += self.energy_sources[i]["color"][0]
            self.target_color[1] += self.energy_sources[i]["color"][1]
            self.target_color[2] += self.energy_sources[i]["color"][2]
        self.target_color = tuple(min(c, 255) for c in self.target_color)

        self._calculate_beam_path()
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False
        action_taken = False

        movement = action[0]

        if self.last_rotated_timer > 0:
            self.last_rotated_timer -= 1
        else:
            self.last_rotated_mirror = -1

        if movement > 0: # 0 is no-op
            mirror_idx = movement - 1
            if mirror_idx < len(self.mirrors) and self.mirrors[mirror_idx]["controllable"]:
                # --- Action SFX Placeholder ---
                # play_sound("mirror_rotate")
                self.mirrors[mirror_idx]["angle"] = (self.mirrors[mirror_idx]["angle"] + 1) % 4
                self.turns_left -= 1
                action_taken = True
                self.last_rotated_mirror = mirror_idx
                self.last_rotated_timer = 5 # Highlight for 5 frames

        if action_taken:
            newly_collected_count = self._calculate_beam_path()
            reward += newly_collected_count * 0.1
            self.score += newly_collected_count * 1

            if self._check_win_condition():
                # --- Win SFX Placeholder ---
                # play_sound("win_jingle")
                reward += 100.0
                self.score += 100
                self.win_state = True
                terminated = True

        if self.turns_left <= 0 and not terminated:
            # --- Lose SFX Placeholder ---
            # play_sound("lose_buzzer")
            terminated = True

        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        self.game_over = terminated
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_beam_path(self):
        for source in self.energy_sources:
            source["collected"] = False

        pos = np.array(self.BEAM_ORIGIN_CELL, dtype=float) + 0.5
        direction = np.array(self.BEAM_INITIAL_DIR, dtype=float)
        self.beam_color = list(self.BEAM_INITIAL_COLOR)
        self.beam_path = [pos.copy()]
        newly_collected_count = 0

        for _ in range(self.GRID_W * self.GRID_H): # Max path length
            grid_pos = tuple(int(p) for p in pos)

            # Check for energy source collection
            for source in self.energy_sources:
                if source["pos"] == grid_pos and not source["collected"]:
                    # --- Collect SFX Placeholder ---
                    # play_sound("collect_chime")
                    source["collected"] = True
                    newly_collected_count += 1
                    for i in range(3):
                        self.beam_color[i] = min(255, self.beam_color[i] + source["color"][i])

            # Check for mirror collision
            hit_mirror = False
            for mirror in self.mirrors:
                if mirror["pos"] == grid_pos:
                    # --- Reflect SFX Placeholder ---
                    # play_sound("beam_reflect")
                    angle_type = mirror["angle"] % 2 # 0 for \, 1 for /
                    
                    if angle_type == 0: # \ mirror
                        direction = np.array([direction[1], direction[0]])
                    else: # / mirror
                        direction = np.array([-direction[1], -direction[0]])
                    
                    hit_mirror = True
                    break
            
            pos += direction
            self.beam_path.append(pos.copy())

            # Check for wall collision
            if not (0 <= pos[0] < self.GRID_W and 0 <= pos[1] < self.GRID_H):
                break
        
        self.beam_color = tuple(self.beam_color)
        return newly_collected_count

    def _check_win_condition(self):
        color_diff = sum(abs(c1 - c2) for c1, c2 in zip(self.beam_color, self.target_color))
        return color_diff < 10 # Allow for small tolerance

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "turns_left": self.turns_left,
            "win": self.win_state,
        }

    def _render_game(self):
        self._render_grid()
        self._render_energy_sources()
        self._render_beam()
        self._render_mirrors()

    def _render_grid(self):
        for x in range(0, self.WIDTH, self.GRID_CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_energy_sources(self):
        radius = self.GRID_CELL_SIZE // 4
        for source in self.energy_sources:
            center_x = int(source["pos"][0] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2)
            center_y = int(source["pos"][1] * self.GRID_CELL_SIZE + self.GRID_CELL_SIZE / 2)
            color = source["color"] if not source["collected"] else self.COLOR_GRID
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)

    def _render_mirrors(self):
        margin = self.GRID_CELL_SIZE // 5
        for i, mirror in enumerate(self.mirrors):
            cell_x = mirror["pos"][0] * self.GRID_CELL_SIZE
            cell_y = mirror["pos"][1] * self.GRID_CELL_SIZE
            
            color = self.COLOR_MIRROR
            if i == self.last_rotated_mirror and self.last_rotated_timer > 0:
                color = self.COLOR_MIRROR_ACTIVE
            
            angle_type = mirror["angle"] % 2
            if angle_type == 0: # \
                start = (cell_x + margin, cell_y + margin)
                end = (cell_x + self.GRID_CELL_SIZE - margin, cell_y + self.GRID_CELL_SIZE - margin)
            else: # /
                start = (cell_x + margin, cell_y + self.GRID_CELL_SIZE - margin)
                end = (cell_x + self.GRID_CELL_SIZE - margin, cell_y + margin)

            pygame.draw.line(self.screen, color, start, end, 5)

    def _render_beam(self):
        if len(self.beam_path) < 2:
            return

        for i in range(len(self.beam_path) - 1):
            p1 = self.beam_path[i]
            p2 = self.beam_path[i+1]
            start_pos = (int(p1[0] * self.GRID_CELL_SIZE), int(p1[1] * self.GRID_CELL_SIZE))
            end_pos = (int(p2[0] * self.GRID_CELL_SIZE), int(p2[1] * self.GRID_CELL_SIZE))
            
            # Glow effect
            glow_color = tuple(min(255, c + 50) for c in self.beam_color)
            pygame.draw.line(self.screen, glow_color, start_pos, end_pos, 12)
            pygame.draw.line(self.screen, self.beam_color, start_pos, end_pos, 6)

    def _render_ui(self):
        # Turns Left
        turns_text = self.font.render(f"TURNS: {self.turns_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(turns_text, (20, 10))

        # Target Color
        target_text = self.font_small.render("TARGET", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_text, (self.WIDTH - 120, 10))
        pygame.draw.rect(self.screen, self.target_color, (self.WIDTH - 120, 35, 100, 30))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (self.WIDTH - 120, 35, 100, 30), 1)

        # Current Beam Color
        current_text = self.font_small.render("CURRENT", True, self.COLOR_UI_TEXT)
        self.screen.blit(current_text, (self.WIDTH - 120, 75))
        pygame.draw.rect(self.screen, self.beam_color, (self.WIDTH - 120, 100, 100, 30))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (self.WIDTH - 120, 100, 100, 30), 1)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "SUCCESS!" if self.win_state else "OUT OF TURNS"
            end_text = self.font.render(end_text_str, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This example requires a display. If you are running headless, this will not work.
    # To run with a display, comment out `os.environ.setdefault("SDL_VIDEODRIVER", "dummy")`
    # at the top of the file.
    
    # Check if a display is available before trying to create one
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Running in headless mode. Manual play example is disabled.")
        print("To run with a display, comment out the SDL_VIDEODRIVER line and ensure you have a display environment.")
        
        # A simple non-interactive test
        print("\nRunning a simple non-interactive test...")
        env = GameEnv()
        obs, info = env.reset()
        print(f"Initial info: {info}")
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {term}, Info: {info}")
            if term:
                print("Game ended. Resetting.")
                obs, info = env.reset()
        env.close()
        print("Non-interactive test finished.")
        exit()


    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("ChromaShift")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n--- ChromaShift Manual Control ---")
    print("Keys 1-4: Rotate mirrors 1-4")
    print("Key 0: No-op (skip turn)")
    print("R: Reset environment")
    print("Q: Quit")
    
    running = True
    while running:
        action = None # Wait for a key press
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    print("--- Environment Reset ---")
                
                if not terminated:
                    if pygame.K_1 <= event.key <= pygame.K_4:
                        action = [event.key - pygame.K_0, 0, 0]
                    elif event.key == pygame.K_0:
                        action = [0, 0, 0]

        if action is not None and not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: Rotate Mirror {action[0]}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Turns Left: {info['turns_left']}")
            if terminated:
                print(f"Game Over! Final Score: {info['score']}. Win: {info['win']}")
        
        # Render the observation from the environment
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()
    pygame.quit()