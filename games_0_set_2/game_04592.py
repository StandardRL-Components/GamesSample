
# Generated: 2025-08-28T02:52:43.282350
# Source Brief: brief_04592.md
# Brief Index: 4592

        
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
        "Controls: Use Left, Down, Up, Right arrow keys (or A, S, W, D) "
        "to hit the notes in the corresponding lanes as they reach the bottom."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced rhythm game. Hit the falling notes in time with the music "
        "to complete all the songs."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 7500  # Allows for ~4 minutes of gameplay
        self.NUM_LANES = 4
        self.NUM_SONGS = 5
        self.MAX_MISSES = 5
        
        # Visual constants
        self.LANE_WIDTH = 80
        self.GRID_WIDTH = self.NUM_LANES * self.LANE_WIDTH
        self.GRID_START_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.NOTE_RADIUS = 25
        self.HIT_ZONE_HEIGHT = 15
        self.HIT_ZONE_Y = self.HEIGHT - 80
        self.EFFECT_MAX_RADIUS = 40
        self.EFFECT_DURATION = 10 # frames
        
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_LANE_BG = (25, 30, 50)
        self.COLOR_LANE_LINE = (50, 60, 90)
        self.COLOR_TARGET_ZONE = (0, 150, 255)
        self.COLOR_NOTE = (255, 255, 255)
        self.COLOR_HIT = (0, 255, 150)
        self.COLOR_MISS = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.LANE_COLORS = [
            (255, 80, 80),   # Left (A)
            (80, 255, 80),   # Down (S)
            (80, 80, 255),   # Up (W)
            (255, 255, 80),  # Right (D)
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        self.notes = []
        self.feedback_effects = []
        self.song_definitions = []
        self.current_song_index = 0
        self.current_song_note_index = 0
        self.spawn_timer = 0
        self.note_speed = 0
        self.notes_in_song = 0
        self.notes_cleared_in_song = 0
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def _generate_songs(self):
        self.song_definitions = []
        base_notes = 20
        base_delay = 45 # frames
        
        for i in range(self.NUM_SONGS):
            song = []
            num_notes = base_notes + i * 5
            min_delay = max(15, base_delay - i * 5)
            
            for _ in range(num_notes):
                delay = self.np_random.integers(min_delay, min_delay + 20)
                lane_action_map = [3, 2, 1, 4] # Left, Down, Up, Right
                lane_type = self.np_random.choice(lane_action_map)
                song.append((delay, lane_type))
            self.song_definitions.append(song)
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_songs()
        
        self.steps = 0
        self.score = 0
        self.misses = 0
        self.game_over = False
        
        self.notes = []
        self.feedback_effects = []
        
        self._load_song(0)
        
        return self._get_observation(), self._get_info()

    def _load_song(self, song_index):
        self.current_song_index = song_index
        self.current_song_note_index = 0
        self.notes_in_song = len(self.song_definitions[song_index])
        self.notes_cleared_in_song = 0
        
        if self.notes_in_song > 0:
            self.spawn_timer = self.song_definitions[song_index][0][0]
        else:
            self.spawn_timer = -1 # No notes to spawn

        base_speed = 3.0
        speed_increment = 0.75
        self.note_speed = base_speed + song_index * speed_increment

    def step(self, action):
        reward = 0
        terminated = False
        
        self._spawn_notes()
        self._update_notes()
        self._update_effects()

        # Unpack factorized action
        # 0=none, 1=up, 2=down, 3=left, 4=right
        key_press = action[0]

        hit_reward = self._process_input(key_press)
        reward += hit_reward
        
        miss_reward = self._process_misses()
        reward += miss_reward

        completion_reward, game_won = self._check_song_completion()
        reward += completion_reward
        
        self.steps += 1
        
        if self.misses >= self.MAX_MISSES:
            terminated = True
            reward -= 100
            # SFX: Game over failure sound
        elif game_won:
            terminated = True
            reward += 100
            # SFX: Game won fanfare
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_notes(self):
        if self.spawn_timer > 0:
            self.spawn_timer -= 1
        
        if self.spawn_timer == 0 and self.current_song_note_index < self.notes_in_song:
            delay, note_type = self.song_definitions[self.current_song_index][self.current_song_note_index]
            
            # Map action type to lane index: 3->0, 2->1, 1->2, 4->3
            lane_map = {3: 0, 2: 1, 1: 2, 4: 3}
            lane_index = lane_map[note_type]
            
            x_pos = self.GRID_START_X + lane_index * self.LANE_WIDTH + self.LANE_WIDTH // 2
            
            new_note = {
                "x": x_pos,
                "y": -self.NOTE_RADIUS,
                "type": note_type,
                "lane": lane_index
            }
            self.notes.append(new_note)
            
            self.current_song_note_index += 1
            if self.current_song_note_index < self.notes_in_song:
                self.spawn_timer = self.song_definitions[self.current_song_index][self.current_song_note_index][0]
            else:
                self.spawn_timer = -1 # All notes for this song spawned

    def _update_notes(self):
        for note in self.notes:
            note["y"] += self.note_speed

    def _update_effects(self):
        for effect in self.feedback_effects[:]:
            effect["timer"] -= 1
            if effect["timer"] <= 0:
                self.feedback_effects.remove(effect)

    def _process_input(self, key_press):
        if key_press == 0:
            return 0

        hit_zone_top = self.HIT_ZONE_Y - self.HIT_ZONE_HEIGHT
        hit_zone_bottom = self.HIT_ZONE_Y + self.HIT_ZONE_HEIGHT
        
        # Find the note closest to the hit line in the correct lane
        best_target = None
        min_dist = float('inf')

        for note in self.notes:
            if note["type"] == key_press:
                if hit_zone_top <= note["y"] <= hit_zone_bottom:
                    dist = abs(note["y"] - self.HIT_ZONE_Y)
                    if dist < min_dist:
                        min_dist = dist
                        best_target = note
        
        if best_target:
            self.notes.remove(best_target)
            self.score += 1
            self.notes_cleared_in_song += 1
            self._add_feedback_effect(best_target["x"], self.HIT_ZONE_Y, self.COLOR_HIT)
            # SFX: Note hit success
            return 1
        
        return 0 # No reward if key press hits nothing

    def _process_misses(self):
        reward = 0
        miss_line = self.HIT_ZONE_Y + self.HIT_ZONE_HEIGHT
        for note in self.notes[:]:
            if note["y"] > miss_line:
                self.notes.remove(note)
                self.misses += 1
                self.notes_cleared_in_song += 1
                reward -= 1
                self._add_feedback_effect(note["x"], self.HIT_ZONE_Y, self.COLOR_MISS)
                # SFX: Note miss sound
        return reward

    def _check_song_completion(self):
        if self.notes_cleared_in_song >= self.notes_in_song and self.notes_in_song > 0:
            if self.current_song_index + 1 >= self.NUM_SONGS:
                return 5, True  # Game won
            else:
                self._load_song(self.current_song_index + 1)
                # SFX: Song complete fanfare
                return 5, False # Song completed, game continues
        return 0, False

    def _add_feedback_effect(self, x, y, color):
        self.feedback_effects.append({
            "x": x, "y": y, "color": color, "timer": self.EFFECT_DURATION
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw lanes
        for i in range(self.NUM_LANES + 1):
            x = self.GRID_START_X + i * self.LANE_WIDTH
            pygame.draw.line(self.screen, self.COLOR_LANE_LINE, (x, 0), (x, self.HEIGHT), 2)
        
        # Draw target zones
        for i in range(self.NUM_LANES):
            lane_x = self.GRID_START_X + i * self.LANE_WIDTH
            target_rect = pygame.Rect(lane_x, self.HIT_ZONE_Y - self.HIT_ZONE_HEIGHT // 2, self.LANE_WIDTH, self.HIT_ZONE_HEIGHT)
            
            # Highlight if a note is inside
            is_note_in_zone = any(
                note['lane'] == i and abs(note['y'] - self.HIT_ZONE_Y) < self.HIT_ZONE_HEIGHT * 2
                for note in self.notes
            )
            color = self.LANE_COLORS[i] if is_note_in_zone else self.COLOR_TARGET_ZONE
            alpha = 100 if is_note_in_zone else 50
            
            s = pygame.Surface((self.LANE_WIDTH, self.HIT_ZONE_HEIGHT), pygame.SRCALPHA)
            s.fill((*color, alpha))
            self.screen.blit(s, (lane_x, self.HIT_ZONE_Y - self.HIT_ZONE_HEIGHT // 2))

            pygame.draw.rect(self.screen, color, target_rect, 2, border_radius=5)

        # Draw feedback effects
        for effect in self.feedback_effects:
            progress = (self.EFFECT_DURATION - effect["timer"]) / self.EFFECT_DURATION
            radius = int(self.EFFECT_MAX_RADIUS * progress)
            alpha = int(255 * (1 - progress))
            color = (*effect["color"], alpha)
            
            # Using gfxdraw for anti-aliasing
            temp_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(temp_surface, radius, radius, radius, color)
            pygame.gfxdraw.filled_circle(temp_surface, radius, radius, radius, color)
            self.screen.blit(temp_surface, (effect['x'] - radius, effect['y'] - radius))

        # Draw notes
        for note in self.notes:
            color = self.LANE_COLORS[note['lane']]
            x, y = int(note["x"]), int(note["y"])
            pygame.gfxdraw.aacircle(self.screen, x, y, self.NOTE_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.NOTE_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.NOTE_RADIUS-5, self.COLOR_BG)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.NOTE_RADIUS-5, self.COLOR_BG)
            pygame.gfxdraw.aacircle(self.screen, x, y, self.NOTE_RADIUS-8, color)
            pygame.gfxdraw.filled_circle(self.screen, x, y, self.NOTE_RADIUS-8, color)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)
        
        # Misses
        miss_text = self.font_medium.render(f"Misses: {self.misses} / {self.MAX_MISSES}", True, self.COLOR_MISS)
        miss_rect = miss_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - 25))
        self.screen.blit(miss_text, miss_rect)
        
        # Song progress
        song_str = f"Song: {self.current_song_index + 1} / {self.NUM_SONGS}"
        song_text = self.font_small.render(song_str, True, self.COLOR_TEXT)
        song_rect = song_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(song_text, song_rect)
        
        # Lane labels
        lane_labels = ["A", "S", "W", "D"]
        lane_keys = ["←", "↓", "↑", "→"]
        action_map = {3: 0, 2: 1, 1: 2, 4: 3} # Maps action to lane index

        for i in range(self.NUM_LANES):
            x = self.GRID_START_X + i * self.LANE_WIDTH + self.LANE_WIDTH // 2
            y = self.HIT_ZONE_Y
            
            key_text = self.font_medium.render(lane_keys[i], True, self.COLOR_TEXT)
            key_rect = key_text.get_rect(center=(x, y))
            self.screen.blit(key_text, key_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "misses": self.misses,
            "song": self.current_song_index + 1
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=42)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=random.randint(0, 1e6))
    
    # Override screen for direct rendering
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption(env.game_description)

    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        action = [0, 0, 0] # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action[0] = 1 # Up
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action[0] = 2 # Down
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action[0] = 3 # Left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action[0] = 4 # Right
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        env.screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.FPS)

    print(f"Game Over!")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward}")
    print(f"Total Steps: {info['steps']}")

    env.close()