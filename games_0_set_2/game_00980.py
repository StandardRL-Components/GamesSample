
# Generated: 2025-08-27T15:24:17.463347
# Source Brief: brief_00980.md
# Brief Index: 980

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ← and → arrow keys to hit notes in the left/right lanes. "
        "Press Space to hit notes in the center lane. Timing is everything!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm game. Hit the descending notes on the beat to score points. "
        "Clear three stages with high accuracy to win."
    )

    # Frames auto-advance for smooth, time-based gameplay.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60 # Run game logic at 60fps for smoother visuals

    # Colors (Catppuccin Macchiato palette)
    COLOR_BG = (24, 25, 38)
    COLOR_GRID = (48, 52, 70)
    COLOR_TEXT = (202, 211, 245)
    COLOR_SUBTLE_TEXT = (136, 140, 158)
    COLOR_TARGET_LINE = (137, 220, 235)
    COLOR_NOTE_COLS = [(245, 194, 231), (203, 166, 247), (148, 226, 213)] # Pink, Mauve, Teal
    COLOR_HIT = (166, 227, 161)
    COLOR_MISS = (243, 139, 168)
    COLOR_TAP_FEEDBACK = (249, 226, 175)

    # Game parameters
    NUM_COLS = 3
    COL_WIDTH = 80
    TARGET_LINE_Y = 320
    HIT_WINDOW = 25  # pixels above/below target line
    
    STAGE_DURATION_SECONDS = 60
    MAX_STEPS = STAGE_DURATION_SECONDS * 3 * FPS
    
    BASE_BPM = 130
    BASE_NOTE_SPEED = 2.5
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # Action space mapping:
        # action[0]: 3=Left, 4=Right. Others are no-ops for tapping.
        # action[1]: 1=Center Tap
        # action[2]: Unused
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_subtitle = pygame.font.SysFont("Consolas", 16)
        self.font_game_over = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        self.np_random = None
        self.notes = []
        self.particles = []
        self.feedback_fx = []
        self.tap_feedback = []

        self.reset()
        
        # Run self-check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        self.stage = 1
        self.stage_timer = 0.0
        
        self.notes = []
        self.particles = []
        self.feedback_fx = []
        self.tap_feedback = []

        self.total_notes_spawned = [0, 0, 0] # per stage
        self.notes_hit = [0, 0, 0]
        self.notes_missed = [0, 0, 0]
        
        self.beat_interval = 60.0 / self.BASE_BPM
        self.time_since_last_beat = 0.0
        self.note_speed = self.BASE_NOTE_SPEED
        
        self.current_reward = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.current_reward = 0
        
        # --- Game Logic Update ---
        self._update_time()
        self._update_stage()
        self._spawn_notes()
        self._process_actions(action)
        self._update_notes()
        self._update_effects()
        
        self.steps += 1
        terminated = self._check_termination()
        
        reward = self.current_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _update_time(self):
        # Use a fixed delta time for deterministic physics
        dt = 1.0 / self.FPS
        self.stage_timer += dt
        self.time_since_last_beat += dt

    def _update_stage(self):
        if self.stage_timer >= self.STAGE_DURATION_SECONDS:
            # Check stage failure condition
            total_notes = self.total_notes_spawned[self.stage - 1]
            missed_notes = self.notes_missed[self.stage - 1]
            if total_notes > 0 and (missed_notes / total_notes) > 0.2:
                self.game_over = True
                self.win_state = False
                return

            if self.stage < 3:
                self.stage += 1
                self.stage_timer = 0
                self.note_speed = self.BASE_NOTE_SPEED + (self.stage - 1) * 0.25 # Slight speed increase
                self.beat_interval = 60.0 / (self.BASE_BPM + (self.stage - 1) * 10)
                self.current_reward += 5 # Stage completion bonus
            else:
                # Game won
                self.game_over = True
                self.win_state = True
                self.current_reward += 50 # Final win bonus

    def _spawn_notes(self):
        if self.time_since_last_beat >= self.beat_interval:
            self.time_since_last_beat -= self.beat_interval
            # 80% chance to spawn a note on a beat
            if self.np_random.random() < 0.80 and not self.game_over:
                col = self.np_random.integers(0, self.NUM_COLS)
                self.notes.append({
                    "col": col,
                    "y": -20,
                    "color": self.COLOR_NOTE_COLS[col],
                    "hit": False
                })
                self.total_notes_spawned[self.stage - 1] += 1
    
    def _process_actions(self, action):
        movement, space_held, _ = action
        
        tapped_cols = []
        if movement == 3: tapped_cols.append(0) # Left
        if movement == 4: tapped_cols.append(2) # Right
        if space_held == 1: tapped_cols.append(1) # Center

        for col in tapped_cols:
            self.tap_feedback.append({"col": col, "alpha": 255})
            # Find a note to hit in this column
            best_note = None
            min_dist = float('inf')
            for note in self.notes:
                if not note["hit"] and note["col"] == col:
                    dist = abs(note["y"] - self.TARGET_LINE_Y)
                    if dist < self.HIT_WINDOW and dist < min_dist:
                        best_note = note
                        min_dist = dist
            
            if best_note:
                best_note["hit"] = True
                self.current_reward += 1
                self.notes_hit[self.stage - 1] += 1
                self._create_hit_fx(col)
                # sfx: positive hit sound

    def _update_notes(self):
        for note in self.notes[:]:
            note["y"] += self.note_speed
            if note["y"] > self.TARGET_LINE_Y + self.HIT_WINDOW and not note["hit"]:
                note["hit"] = True # Mark as handled
                self.current_reward -= 1
                self.notes_missed[self.stage - 1] += 1
                self._create_miss_fx(note["col"])
                # sfx: miss sound
        
        # Remove handled notes that are off-screen
        self.notes = [n for n in self.notes if n["y"] < self.SCREEN_HEIGHT]

    def _update_effects(self):
        # Particles
        for p in self.particles[:]:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["alpha"] -= p["fade"]
            if p["alpha"] <= 0:
                self.particles.remove(p)

        # Hit/Miss text feedback
        for fx in self.feedback_fx[:]:
            fx["y"] -= 0.5
            fx["alpha"] -= 4
            if fx["alpha"] <= 0:
                self.feedback_fx.remove(fx)
                
        # Tap visual feedback
        for tf in self.tap_feedback[:]:
            tf["alpha"] -= 25
            if tf["alpha"] <= 0:
                self.tap_feedback.remove(tf)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        
        # Check running failure condition
        total_notes = self.total_notes_spawned[self.stage - 1]
        missed_notes = self.notes_missed[self.stage - 1]
        if total_notes > 10 and (missed_notes / total_notes) > 0.35: # More lenient mid-stage
            self.game_over = True
            self.win_state = False
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        current_stage_idx = self.stage - 1
        total_notes = self.total_notes_spawned[current_stage_idx]
        hits = self.notes_hit[current_stage_idx]
        accuracy = (hits / total_notes) * 100 if total_notes > 0 else 100.0
        
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "stage_accuracy": accuracy
        }

    def _render_game(self):
        grid_center_x = self.SCREEN_WIDTH // 2
        total_grid_width = self.NUM_COLS * self.COL_WIDTH
        grid_start_x = grid_center_x - total_grid_width // 2

        # Draw column tap feedback
        for tf in self.tap_feedback:
            x = grid_start_x + tf["col"] * self.COL_WIDTH
            s = pygame.Surface((self.COL_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            color = (*self.COLOR_TAP_FEEDBACK, tf["alpha"])
            s.fill(color)
            self.screen.blit(s, (x, 0))

        # Draw grid lines
        for i in range(self.NUM_COLS + 1):
            x = grid_start_x + i * self.COL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)

        # Draw target line with glow
        pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (grid_start_x, self.TARGET_LINE_Y), (grid_start_x + total_grid_width, self.TARGET_LINE_Y), 3)
        
        # Draw notes
        for note in self.notes:
            if not note["hit"]:
                note_x = grid_start_x + note["col"] * self.COL_WIDTH
                note_y = int(note["y"])
                note_rect = pygame.Rect(note_x, note_y - 10, self.COL_WIDTH, 20)
                
                # Glow effect for notes near target
                dist_to_target = abs(note_y - self.TARGET_LINE_Y)
                if dist_to_target < 80:
                    alpha = int(max(0, 255 * (1 - dist_to_target / 80)))
                    self._draw_glow(note_rect.center, 25, note["color"], alpha)

                pygame.draw.rect(self.screen, note["color"], note_rect, border_radius=4)
                pygame.draw.rect(self.screen, self.COLOR_BG, note_rect, 2, border_radius=4)

        # Draw particles
        for p in self.particles:
            color = (*p["color"], int(p["alpha"]))
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), int(p["size"]), color)
            
        # Draw feedback fx
        for fx in self.feedback_fx:
            text_surf = fx["font"].render(fx["text"], True, (*fx["color"], fx["alpha"]))
            text_rect = text_surf.get_rect(center=(fx["x"], int(fx["y"])))
            self.screen.blit(text_surf, text_rect)
            
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_text = self.font_main.render(f"STAGE: {self.stage}/3", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(stage_text, stage_rect)

        # Stage progress bar
        progress = self.stage_timer / self.STAGE_DURATION_SECONDS
        bar_width = 200
        bar_height = 10
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        bar_y = 15
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TARGET_LINE, (bar_x, bar_y, bar_width * progress, bar_height), border_radius=5)
        
        # Accuracy
        current_stage_idx = self.stage - 1
        total_notes = self.total_notes_spawned[current_stage_idx]
        hits = self.notes_hit[current_stage_idx]
        accuracy = (hits / total_notes) * 100 if total_notes > 0 else 100.0
        acc_text = self.font_subtitle.render(f"ACC: {accuracy:.1f}%", True, self.COLOR_SUBTLE_TEXT)
        acc_rect = acc_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 40))
        self.screen.blit(acc_text, acc_rect)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state else "GAME OVER"
            color = self.COLOR_HIT if self.win_state else self.COLOR_MISS
            
            end_text = self.font_game_over.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)
    
    def _create_hit_fx(self, col):
        grid_center_x = self.SCREEN_WIDTH // 2
        total_grid_width = self.NUM_COLS * self.COL_WIDTH
        grid_start_x = grid_center_x - total_grid_width // 2
        x = grid_start_x + col * self.COL_WIDTH + self.COL_WIDTH / 2
        y = self.TARGET_LINE_Y
        
        # Add score text feedback
        self.score += 10
        self.feedback_fx.append({
            "text": "PERFECT", "x": x, "y": y - 10, "alpha": 255, 
            "color": self.COLOR_HIT, "font": self.font_subtitle
        })
        
        # Create particles
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "size": self.np_random.uniform(1, 4),
                "color": self.COLOR_HIT,
                "alpha": 255,
                "fade": self.np_random.uniform(3, 6)
            })

    def _create_miss_fx(self, col):
        grid_center_x = self.SCREEN_WIDTH // 2
        total_grid_width = self.NUM_COLS * self.COL_WIDTH
        grid_start_x = grid_center_x - total_grid_width // 2
        x = grid_start_x + col * self.COL_WIDTH + self.COL_WIDTH / 2
        
        self.feedback_fx.append({
            "text": "MISS", "x": x, "y": self.TARGET_LINE_Y - 10, "alpha": 255,
            "color": self.COLOR_MISS, "font": self.font_main
        })

    def _draw_glow(self, center, radius, color, alpha):
        if alpha <= 0: return
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        glow_color = (*color, int(alpha))
        pygame.draw.circle(s, glow_color, (radius, radius), radius)
        self.screen.blit(s, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run with manual control ---
    # This requires a display and is not part of the core headless requirement,
    # but is useful for testing and visualization.
    try:
        import sys
        
        # Re-initialize pygame with a display
        pygame.display.init()
        pygame.display.set_caption("Rhythm Game")
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        obs, info = env.reset()
        terminated = False
        
        print("\n" + "="*30)
        print("MANUAL CONTROL MODE")
        print(env.user_guide)
        print("="*30 + "\n")

        while not terminated and env.steps < env.MAX_STEPS:
            # Action defaults
            movement = 0
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: movement = 3
            if keys[pygame.K_RIGHT]: movement = 4
            if keys[pygame.K_SPACE]: space = 1
            # Shift is not used in this game
            
            action = np.array([movement, space, shift])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}, Accuracy: {info['stage_accuracy']:.1f}%")

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(env.FPS)

        print("Game Over!")
        print(f"Final Score: {info['score']}")

    except ImportError:
        print("Pygame display not available. Cannot run manual control test.")
    except pygame.error as e:
        print(f"Pygame error (likely no display available): {e}")
        print("Cannot run manual control test. The environment is headless-compatible.")
    finally:
        env.close()