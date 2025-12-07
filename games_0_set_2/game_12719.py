import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:20:26.249243
# Source Brief: brief_02719.md
# Brief Index: 2719
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Fractal Navigator Game Environment

    In this puzzle game, the agent must manipulate a procedurally generated
    fractal structure to create a continuous path for a player entity to
    reach an exit portal. The agent controls which fractal segment is
    selected, can rotate it, and can toggle its solidity.

    The game is time-limited, and the agent must solve 3 levels of
    increasing complexity to win.
    """
    game_description = (
        "Navigate a procedurally generated fractal by rotating and phasing segments to create a path for your energy core to reach the exit portal before time runs out."
    )
    user_guide = (
        "Controls: Use ↑↓ arrow keys to rotate the selected fractal segment and ←→ to select the next or previous segment. Press space to toggle a segment's solidity."
    )
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.MAX_LEVELS = 3
        self.TOTAL_GAME_SECONDS = 60
        self.MAX_EPISODE_STEPS = self.TOTAL_GAME_SECONDS * self.FPS + 100 # Safety buffer

        # --- Colors ---
        self.COLOR_BG_START = (5, 10, 25)
        self.COLOR_BG_END = (0, 0, 0)
        self.COLOR_FRACTAL_SOLID = (0, 255, 255) # Cyan
        self.COLOR_FRACTAL_GHOST = (255, 255, 255, 60)
        self.COLOR_SELECT_GLOW = (255, 255, 0, 100) # Yellow glow
        self.COLOR_PLAYER = (50, 255, 100) # Bright Green
        self.COLOR_EXIT = (255, 200, 0) # Gold/Yellow
        self.COLOR_TEXT = (220, 220, 220)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        self._create_background()

        # --- Game State Variables ---
        self.current_level = 0
        self.steps_in_episode = 0
        self.total_game_time_steps = 0
        self.score = 0.0
        self.game_over = False
        self.prev_space_held = False
        self.action_cooldown = 0
        self.fractal_segments = []
        self.selected_segment_idx = 0
        self.fractal_origin = (0, 0)
        self.player_pos = np.array([0.0, 0.0])
        self.player_target_pos = np.array([0.0, 0.0])
        self.exit_pos = np.array([0.0, 0.0])
        self.last_dist_to_exit = 0.0

        # Initialize state by calling reset
        # self.reset() is called by the environment wrapper
        
    def _create_background(self):
        """Creates a pre-rendered background surface for efficiency."""
        self.background_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            # Interpolate color from top to bottom
            interp = y / self.SCREEN_HEIGHT
            color = [
                int(self.COLOR_BG_START[i] * (1 - interp) + self.COLOR_BG_END[i] * interp)
                for i in range(3)
            ]
            pygame.draw.line(self.background_surf, color, (0, y), (self.SCREEN_WIDTH, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_level = 1
        self.steps_in_episode = 0
        self.total_game_time_steps = self.TOTAL_GAME_SECONDS * self.FPS
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.action_cooldown = 0
        
        self._generate_level()
        self.last_dist_to_exit = self._get_dist_to_exit()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Procedurally generates the fractal for the current level."""
        # Sound: level_start.wav
        num_segments = 1 + 2 * self.current_level # 3, 5, 7
        self.fractal_segments = []
        base_length = 120 - self.current_level * 15 # Shorter segments for more complex fractals

        for i in range(num_segments):
            self.fractal_segments.append({
                'start': np.array([0.0, 0.0]),
                'end': np.array([0.0, 0.0]),
                'angle': self.np_random.choice([0, 90, 180, 270]),
                'length': base_length * (0.9 ** i),
                'solid': self.np_random.random() > 0.2 # 80% chance to start solid
            })
        
        self.selected_segment_idx = 0
        self.fractal_origin = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        self._recalculate_fractal_geometry()
        self._update_path()
        self.player_pos = np.copy(self.player_target_pos)

    def _recalculate_fractal_geometry(self):
        """Updates the start/end coordinates of all segments based on their angles."""
        current_pos = np.array(self.fractal_origin, dtype=float)
        for seg in self.fractal_segments:
            seg['start'] = np.copy(current_pos)
            rad = math.radians(seg['angle'])
            end_offset = np.array([seg['length'] * math.cos(rad), seg['length'] * math.sin(rad)])
            seg['end'] = current_pos + end_offset
            current_pos = seg['end']
        
        self.exit_pos = self.fractal_segments[-1]['end'] if self.fractal_segments else self.fractal_origin

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps_in_episode += 1
        self.total_game_time_steps -= 1

        self._handle_actions(action)
        self._recalculate_fractal_geometry()
        self._update_path()
        self._update_player_movement()

        reward, level_completed = self._calculate_reward()
        self.score += reward
        
        if level_completed:
            self.current_level += 1
            if self.current_level > self.MAX_LEVELS:
                # Game won
                pass # Termination check will handle this
            else:
                self._generate_level()
                self.last_dist_to_exit = self._get_dist_to_exit()

        terminated = self._check_termination()
        truncated = False # This game does not truncate
        if terminated:
            self.game_over = True
            # Apply large terminal rewards
            if self.current_level > self.MAX_LEVELS:
                final_reward = 50.0
                self.score += final_reward
                reward += final_reward
            elif self.total_game_time_steps <= 0:
                final_reward = -100.0
                self.score += final_reward
                reward += final_reward

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        selected_seg = self.fractal_segments[self.selected_segment_idx]

        # Rotation (Up/Down)
        if movement == 1: # Clockwise
            selected_seg['angle'] = (selected_seg['angle'] + 90) % 360
            # Sound: rotate_click.wav
        elif movement == 2: # Counter-Clockwise
            selected_seg['angle'] = (selected_seg['angle'] - 90 + 360) % 360
            # Sound: rotate_click.wav

        # Selection (Left/Right) - with cooldown
        if self.action_cooldown == 0:
            if movement == 3: # Previous
                self.selected_segment_idx = (self.selected_segment_idx - 1) % len(self.fractal_segments)
                self.action_cooldown = 5 # 5-frame cooldown
                # Sound: select_ui.wav
            elif movement == 4: # Next
                self.selected_segment_idx = (self.selected_segment_idx + 1) % len(self.fractal_segments)
                self.action_cooldown = 5
                # Sound: select_ui.wav

        # Transparency Toggle (Space) - on rising edge
        if space_held and not self.prev_space_held:
            selected_seg['solid'] = not selected_seg['solid']
            # Sound: toggle_phase.wav
        
        self.prev_space_held = space_held

    def _update_path(self):
        """Finds the furthest reachable point for the player along the solid, connected path."""
        path_end_pos = self.fractal_segments[0]['start']
        
        # Check first segment
        if not self.fractal_segments[0]['solid']:
            self.player_target_pos = path_end_pos
            return
        path_end_pos = self.fractal_segments[0]['end']

        # Check subsequent segments
        for i in range(1, len(self.fractal_segments)):
            current_seg = self.fractal_segments[i]
            prev_seg = self.fractal_segments[i-1]
            
            # Path breaks if segment is not solid or not connected
            dist = np.linalg.norm(prev_seg['end'] - current_seg['start'])
            if not current_seg['solid'] or dist > 1.0: # Tolerance for float precision
                break
            
            path_end_pos = current_seg['end']

        self.player_target_pos = path_end_pos

    def _update_player_movement(self):
        """Smoothly interpolates the player's position towards its target."""
        lerp_factor = 0.2
        self.player_pos += (self.player_target_pos - self.player_pos) * lerp_factor

    def _get_dist_to_exit(self):
        return np.linalg.norm(self.player_target_pos - self.exit_pos)

    def _calculate_reward(self):
        """Calculates reward based on progress and level completion."""
        reward = 0.0
        level_completed = False

        # Reward for making progress
        dist = self._get_dist_to_exit()
        if dist < self.last_dist_to_exit:
            reward += 0.1
        self.last_dist_to_exit = dist

        # Reward for completing a level
        if np.linalg.norm(self.player_pos - self.exit_pos) < 15.0:
            reward += 5.0
            level_completed = True
            # Sound: level_complete.wav

        return reward, level_completed

    def _check_termination(self):
        win = self.current_level > self.MAX_LEVELS
        timeout = self.total_game_time_steps <= 0
        max_steps = self.steps_in_episode >= self.MAX_EPISODE_STEPS
        return win or timeout or max_steps

    def _get_observation(self):
        self.screen.blit(self.background_surf, (0, 0))
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render fractal segments
        for i, seg in enumerate(self.fractal_segments):
            start_pos = tuple(map(int, seg['start']))
            end_pos = tuple(map(int, seg['end']))
            
            if i == self.selected_segment_idx:
                # Draw a glow effect for the selected segment
                pygame.draw.line(self.screen, self.COLOR_SELECT_GLOW, start_pos, end_pos, 12)

            if seg['solid']:
                pygame.draw.aaline(self.screen, self.COLOR_FRACTAL_SOLID, start_pos, end_pos, 3)
            else:
                # Draw ghosted line using gfxdraw for alpha blending
                pygame.draw.aaline(self.screen, self.COLOR_FRACTAL_GHOST, start_pos, end_pos, 2)

        # Render Exit Portal (pulsating)
        pulse = (math.sin(self.steps_in_episode * 0.1) + 1) / 2 # Varies between 0 and 1
        exit_radius = int(10 + pulse * 5)
        exit_center = tuple(map(int, self.exit_pos))
        pygame.gfxdraw.filled_circle(self.screen, exit_center[0], exit_center[1], exit_radius, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, exit_center[0], exit_center[1], exit_radius, self.COLOR_EXIT)

        # Render Player (pulsating)
        player_radius = int(6 + pulse * 2)
        player_center = tuple(map(int, self.player_pos))
        pygame.gfxdraw.filled_circle(self.screen, player_center[0], player_center[1], player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_center[0], player_center[1], player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Level Text
        level_text = self.font_small.render(f"Level: {self.current_level}/{self.MAX_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Timer Text
        remaining_seconds = max(0, self.total_game_time_steps // self.FPS)
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        timer_text = self.font_large.render(f"{minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 15, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.current_level > self.MAX_LEVELS:
                msg = "SUCCESS"
            else:
                msg = "TIME OUT"

            end_text = self.font_large.render(msg, True, self.COLOR_EXIT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "level": self.current_level,
            "steps": self.steps_in_episode,
            "time_remaining_steps": self.total_game_time_steps
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Create a display surface
    pygame.display.set_caption("Fractal Navigator")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset | Q: Quit")

    while not done:
        # Default action is "do nothing"
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                
                # Map keys to actions
                if event.key == pygame.K_UP:
                    action[0] = 2 # Rotate CCW (intuitive for up)
                elif event.key == pygame.K_DOWN:
                    action[0] = 1 # Rotate CW (intuitive for down)
                elif event.key == pygame.K_LEFT:
                    action[0] = 3 # Select Prev
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4 # Select Next
                
        # Handle held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Space held

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # For human play, we need to render to the screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()