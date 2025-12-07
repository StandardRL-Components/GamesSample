import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:25:28.577629
# Source Brief: brief_00979.md
# Brief Index: 979
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a rhythm/puzzle game.
    The player controls a cursor on a musical staff, collecting notes on the beat
    to trigger chain reactions and achieve a high score and synchronization.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    game_description = (
        "Collect notes on the beat in this rhythm-puzzle game. Trigger chain reactions to boost your score and maintain synchronization."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to collect a note and start a chain reaction."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_TIME_SECONDS = 45
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.WIN_SCORE = 50
        self.SYNC_LOSS_THRESHOLD = 80.0

        # --- Colors (High Contrast, Visually Appealing) ---
        self.COLOR_BG = (15, 10, 35)
        self.COLOR_STAFF = (50, 40, 80)
        self.COLOR_PLAYER = (0, 180, 255)
        self.COLOR_PLAYER_GLOW = (0, 100, 200)
        self.NOTE_COLORS = [
            (255, 80, 80), (255, 160, 80), (255, 255, 80),
            (80, 255, 80), (80, 255, 255), (80, 80, 255),
            (160, 80, 255), (255, 80, 255), (255, 200, 200)
        ]
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.COLOR_RHYTHM_BAR = (100, 80, 160)
        self.COLOR_RHYTHM_PULSE = (200, 180, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_game_over = pygame.font.Font(None, 52)
        
        # --- Game Configuration ---
        self.staff_y_positions = [self.HEIGHT // 2 + i * 15 for i in range(-4, 5)]
        self.player_speed = 15
        self.bpm = 120
        self.beat_interval_frames = (self.FPS * 60) // self.bpm
        self.beat_window = max(1, self.beat_interval_frames // 10) # +/- frames for a "good" hit

        # --- Initialize State ---
        # These are initialized in reset() to ensure a clean state for each episode
        self.player_pos = None
        self.player_y_index = None
        self.notes = None
        self.particles = None
        self.score = None
        self.steps = None
        self.sync_percentage = None
        self.beat_progress = None
        self.hit_this_beat = None
        self.last_space_state = None
        self.note_spawn_timer = None
        self.note_spawn_rate_sec = None
        self.last_difficulty_increase_step = None
        self.game_over = None
        self.game_result = None # "WIN", "LOSE_TIME", "LOSE_SYNC"
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.WIDTH // 2, self.staff_y_positions[4]]
        self.player_y_index = 4
        self.notes = []
        self.particles = []
        
        self.score = 0
        self.steps = 0
        self.sync_percentage = 100.0
        
        self.beat_progress = 0
        self.hit_this_beat = False
        self.last_space_state = 0
        
        self.note_spawn_timer = 0
        self.note_spawn_rate_sec = 1.5
        self.last_difficulty_increase_step = 0
        
        self.game_over = False
        self.game_result = None

        # Populate initial notes to start the game
        for _ in range(5):
            self._spawn_note()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_state
        self.last_space_state = space_held
        
        self._handle_movement(movement)

        # --- Game Logic Updates ---
        self._update_difficulty()
        self._spawn_notes()
        self._update_particles()
        
        rhythm_reward = self._update_rhythm()
        reward += rhythm_reward
        
        if space_pressed:
            collection_reward = self._collect_notes_at_cursor()
            reward += collection_reward
            # sfx: note_collect.wav or chain_collect.wav

        # --- Termination Check ---
        terminated = False
        terminal_reward = 0
        
        if self.score >= self.WIN_SCORE:
            terminated = True
            self.game_result = "WIN"
            terminal_reward = 50.0
        elif self.sync_percentage < self.SYNC_LOSS_THRESHOLD:
            terminated = True
            self.game_result = "LOSE_SYNC"
            terminal_reward = -50.0
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_result = "LOSE_TIME"
            # No specific reward, the lack of win reward is the penalty

        if terminated:
            self.game_over = True
            reward += terminal_reward

        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated: # Ensure game over is set if truncated
            self.game_over = True
            self.game_result = "LOSE_TIME"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: # Up
            self.player_y_index = max(0, self.player_y_index - 1)
        elif movement == 2: # Down
            self.player_y_index = min(len(self.staff_y_positions) - 1, self.player_y_index + 1)
        elif movement == 3: # Left
            self.player_pos[0] -= self.player_speed
        elif movement == 4: # Right
            self.player_pos[0] += self.player_speed
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = self.staff_y_positions[self.player_y_index]

    def _update_difficulty(self):
        # Increase difficulty every 5 seconds by increasing spawn rate
        if self.steps - self.last_difficulty_increase_step > 5 * self.FPS:
            self.note_spawn_rate_sec = max(0.2, self.note_spawn_rate_sec - 0.05)
            self.last_difficulty_increase_step = self.steps

    def _spawn_notes(self):
        self.note_spawn_timer += 1
        if self.note_spawn_timer >= self.note_spawn_rate_sec * self.FPS:
            self.note_spawn_timer = 0
            self._spawn_note()

    def _spawn_note(self):
        y_index = self.np_random.integers(0, len(self.staff_y_positions))
        note = {
            "pos": [self.np_random.integers(50, self.WIDTH - 50), self.staff_y_positions[y_index]],
            "y_index": y_index,
            "color": self.NOTE_COLORS[y_index],
            "radius": 8,
            "spawn_step": self.steps
        }
        self.notes.append(note)

    def _update_rhythm(self):
        self.beat_progress = (self.beat_progress + 1) % self.beat_interval_frames
        # sfx: metronome_tick.wav on beat
        if self.beat_progress == 0:
            if not self.hit_this_beat:
                # Missed beat penalty
                self.sync_percentage = max(0.0, self.sync_percentage - 2.0)
                # sfx: beat_miss.wav
                return -0.1
            self.hit_this_beat = False
        return 0

    def _collect_notes_at_cursor(self):
        reward = 0
        
        # Find note under cursor
        target_note = None
        for note in self.notes:
            dist = math.hypot(note["pos"][0] - self.player_pos[0], note["pos"][1] - self.player_pos[1])
            if dist < note["radius"] + 7: # 7 is player cursor radius
                target_note = note
                break
        
        if not target_note:
            return 0

        # --- Chain reaction logic (BFS) ---
        notes_to_collect = []
        q = [target_note]
        visited = {id(target_note)}

        while q:
            current_note = q.pop(0)
            notes_to_collect.append(current_note)
            for other_note in self.notes:
                if id(other_note) in visited: continue
                if other_note["y_index"] == current_note["y_index"]:
                    if abs(other_note["pos"][0] - current_note["pos"][0]) < 40:
                        visited.add(id(other_note))
                        q.append(other_note)
        
        # --- Process collection ---
        collected_count = len(notes_to_collect)
        self.score += collected_count
        reward += collected_count * 1.0 # +1 per note

        if collected_count >= 3:
            reward += 5.0 # Chain reaction bonus

        # Check if collection was on beat for sync bonus
        is_on_beat = (self.beat_progress < self.beat_window) or \
                     (self.beat_progress > self.beat_interval_frames - self.beat_window)
        if is_on_beat:
            self.sync_percentage = min(100.0, self.sync_percentage + 1.0 * collected_count)
            self.hit_this_beat = True

        # Create particles and remove notes
        for note in notes_to_collect:
            self._create_particles(note["pos"], note["color"])
            if note in self.notes:
                self.notes.remove(note)

        return reward
        
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos), "vel": vel, "lifetime": self.np_random.integers(20, 40),
                "color": color, "size": self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1
            p["size"] *= 0.95
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sync_percentage": self.sync_percentage,
            "time_remaining_sec": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_rhythm_indicator()
        self._render_particles()
        self._render_notes()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- Rendering Methods ---
    def _render_background(self):
        for y in self.staff_y_positions:
            pygame.draw.line(self.screen, self.COLOR_STAFF, (0, y), (self.WIDTH, y), 2)

    def _render_rhythm_indicator(self):
        bar_height = 10
        bar_y = self.HEIGHT - bar_height
        pygame.draw.rect(self.screen, self.COLOR_RHYTHM_BAR, (0, bar_y, self.WIDTH, bar_height))
        
        pulse_progress = self.beat_progress / self.beat_interval_frames
        pulse_x = int(pulse_progress * self.WIDTH)
        
        pygame.draw.line(self.screen, self.COLOR_RHYTHM_PULSE, (pulse_x, bar_y), (pulse_x, self.HEIGHT), 3)

    def _render_notes(self):
        for note in self.notes:
            pos = (int(note["pos"][0]), int(note["pos"][1]))
            radius = int(note["radius"])
            life = (self.steps - note["spawn_step"]) / 15.0
            current_radius = int(radius * min(1.0, life))
            if current_radius > 1:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], current_radius, note["color"])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], current_radius, note["color"])

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        radius = 7
        for i in range(radius * 2, radius, -2):
            alpha = 40 * (1 - (i / (radius * 2)))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i, (*self.COLOR_PLAYER_GLOW, int(alpha)))
        
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            size = int(p["size"])
            if size > 0:
                alpha = int(255 * (p["lifetime"] / 40.0))
                color = (*p["color"], alpha)
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.rect(s, color, s.get_rect())
                self.screen.blit(s, (pos[0] - size, pos[1] - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        
        score_text = self.font_ui.render(f"Notes: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        time_text = self.font_ui.render(f"Time: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        sync_text = self.font_ui.render(f"Sync: {self.sync_percentage:.1f}%", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))
        self.screen.blit(sync_text, (self.WIDTH // 2 - sync_text.get_width() // 2, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.game_result == "WIN":
            msg = "Synchronization Complete!"
            color = (100, 255, 100)
        elif self.game_result == "LOSE_SYNC":
            msg = "Synchronization Lost!"
            color = (255, 100, 100)
        else: # LOSE_TIME
            msg = "Time Expired"
            color = (255, 200, 100)
        
        text = self.font_game_over.render(msg, True, color)
        text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(text, text_rect)

    def render(self):
        return self._get_observation()
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block will not run in the headless environment, but is useful for local testing.
    # To run it, you might need to comment out the `os.environ.setdefault` line.
    try:
        env = GameEnv()
        obs, info = env.reset()
        done = False
        
        # --- Manual Play Setup ---
        pygame.display.set_caption("Rhythm Sync Environment")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()

        action = [0, 0, 0] # [movement, space, shift]
        
        print("--- Controls ---")
        print("Arrows: Move cursor")
        print("Space: Collect note")
        print("Q: Quit")
        
        running = True
        while running:
            # --- Human Input ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            keys = pygame.key.get_pressed()
            action[0] = 0 # No movement
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
                
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Sync: {info['sync_percentage']:.1f}%")

            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Final Sync: {info['sync_percentage']:.1f}%")
                # Display final frame
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                # Wait a bit before resetting
                pygame.time.wait(3000)
                obs, info = env.reset()

            # --- Rendering ---
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(env.FPS)

        env.close()
    except pygame.error as e:
        print(f"Pygame error (likely due to dummy video driver): {e}")
        print("This is expected in a headless environment. The environment code is likely correct.")
        print("Running validation check...")
        env = GameEnv()
        env.validate_implementation()
        env.close()