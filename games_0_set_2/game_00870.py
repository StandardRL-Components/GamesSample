
# Generated: 2025-08-27T15:02:16.681502
# Source Brief: brief_00870.md
# Brief Index: 870

        
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


class Note:
    """Represents a single note in the game."""
    def __init__(self, note_type, note_map):
        self.type = note_type
        self.x = 640 + 50  # Start off-screen
        self.y = note_map[note_type]['y']
        self.color = note_map[note_type]['color']
        self.shape = note_map[note_type]['shape']
        self.is_hit = False
        self.is_missed = False

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.life = 20  # Lifetime in frames
        self.size = random.randint(4, 8)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys, space, and shift to hit the corresponding colored notes as they cross the hit line."
    )

    game_description = (
        "A retro-futuristic rhythm game. Hit notes on the beat to build your score and clear the song."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.HIT_ZONE_X = 100
        self.HIT_WINDOW = 25  # +/- pixels from HIT_ZONE_X

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_ROAD = (25, 15, 40)
        self.COLOR_LANE = (60, 40, 90)
        self.COLOR_WHITE = (240, 240, 255)
        self.COLOR_GREY = (80, 80, 90)
        self.COLOR_GREEN = (0, 255, 127)
        self.COLOR_RED = (255, 69, 58)
        self.COLOR_BLUE = (0, 122, 255)
        self.COLOR_YELLOW = (255, 204, 0)
        self.COLOR_PURPLE = (175, 82, 222)
        self.COLOR_ORANGE = (255, 149, 0)
        
        # Note mapping
        self.NOTE_MAP = {
            1: {'name': 'Up', 'color': self.COLOR_GREEN, 'y': 80, 'shape': 'rect'},
            2: {'name': 'Down', 'color': self.COLOR_RED, 'y': 320, 'shape': 'rect'},
            3: {'name': 'Left', 'color': self.COLOR_BLUE, 'y': 150, 'shape': 'rect'},
            4: {'name': 'Right', 'color': self.COLOR_YELLOW, 'y': 250, 'shape': 'rect'},
            5: {'name': 'Space', 'color': self.COLOR_PURPLE, 'y': self.HEIGHT // 2, 'shape': 'circle'},
            6: {'name': 'Shift', 'color': self.COLOR_ORANGE, 'y': self.HEIGHT // 2, 'shape': 'diamond'},
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_feedback = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.health = 0.0
        self.game_over = False
        self.notes = []
        self.particles = []
        self.stars = []
        self.song_data = []
        self.song_cursor = 0
        self.note_speed = 0.0
        self.combo = 0
        self.max_combo = 0
        self.notes_hit = 0
        self.notes_missed = 0
        self.last_action_feedback = {}

        self.reset()
        self.validate_implementation()

    def _generate_song(self):
        """Creates a procedural song for the episode."""
        self.song_data = []
        num_notes = 100
        current_step = 50  # Start first note a bit into the song
        
        for _ in range(num_notes):
            note_type = self.np_random.integers(1, 7)
            self.song_data.append({'step': current_step, 'type': note_type})
            
            # Add a second note for a chord sometimes
            if self.np_random.random() < 0.2:
                chord_note_type = self.np_random.integers(1, 7)
                if chord_note_type != note_type:
                     self.song_data.append({'step': current_step, 'type': chord_note_type})

            step_increment = self.np_random.integers(8, 12)
            current_step += step_increment
        
        self.song_data.sort(key=lambda x: x['step'])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.health = 100.0
        self.game_over = False
        self.notes = []
        self.particles = []
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.1, 0.5))
            for _ in range(100)
        ]
        self.note_speed = 3.0
        self.combo = 0
        self.max_combo = 0
        self.notes_hit = 0
        self.notes_missed = 0
        self.song_cursor = 0
        self.last_action_feedback = {k: 0 for k in self.NOTE_MAP.keys()}
        
        self._generate_song()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- Game Logic Update ---
        self.steps += 1
        
        # Difficulty scaling
        self.note_speed = 3.0 + (self.steps // 200) * 0.15

        # Update action feedback timers
        for k in self.last_action_feedback:
            self.last_action_feedback[k] = max(0, self.last_action_feedback[k] - 1)

        # Spawn new notes
        while self.song_cursor < len(self.song_data) and self.steps >= self.song_data[self.song_cursor]['step']:
            note_info = self.song_data[self.song_cursor]
            self.notes.append(Note(note_info['type'], self.NOTE_MAP))
            self.song_cursor += 1

        # Update notes
        notes_to_remove = []
        for note in self.notes:
            note.x -= self.note_speed
            if note.x < self.HIT_ZONE_X - self.HIT_WINDOW and not note.is_hit and not note.is_missed:
                note.is_missed = True
                self.notes_missed += 1
                self.combo = 0
                self.health -= 10
                reward -= 1
                # sfx: miss_sound
            if note.x < -50:
                notes_to_remove.append(note)
        
        # Unpack actions and process hits
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        player_actions = set()
        if movement > 0: player_actions.add(movement)
        if space_held: player_actions.add(5)
        if shift_held: player_actions.add(6)

        for action_type in player_actions:
            self.last_action_feedback[action_type] = 10 # Flash for 10 frames
            for note in self.notes:
                if not note.is_hit and not note.is_missed and note.type == action_type:
                    if abs(note.x - self.HIT_ZONE_X) <= self.HIT_WINDOW:
                        note.is_hit = True
                        self.notes_hit += 1
                        self.score += 1
                        reward += 1
                        self.combo += 1
                        self.max_combo = max(self.max_combo, self.combo)
                        if self.combo > 0 and self.combo % 10 == 0:
                            reward += 5
                        self._create_particles(self.HIT_ZONE_X, note.y, note.color, 20)
                        # sfx: hit_sound
                        break # Only hit one note per action type per frame
        
        self.notes = [n for n in self.notes if n not in notes_to_remove and not n.is_hit]

        # Update particles
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1
        self.particles = [p for p in self.particles if p.life > 0]

        # --- Termination Check ---
        total_notes_processed = self.notes_hit + self.notes_missed
        if self.health <= 0:
            terminated = True
        if self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            accuracy = self.notes_hit / total_notes_processed if total_notes_processed > 0 else 0
            if accuracy >= 0.75 and self.steps >= self.MAX_STEPS:
                reward += 50 # Victory bonus
            else:
                reward -= 50 # Failure penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Pulsating background glow
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        glow_radius = int(150 + pulse * 50)
        glow_alpha = int(20 + pulse * 15)
        temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, self.WIDTH // 2, self.HEIGHT // 2, glow_radius, (*self.COLOR_PURPLE, glow_alpha))
        self.screen.blit(temp_surf, (0, 0))

        # Parallax stars
        for i, (x, y, z) in enumerate(self.stars):
            new_x = (x - self.note_speed * z) % self.WIDTH
            self.stars[i] = (new_x, y, z)
            size = int(z * 3)
            color_val = int(100 + z * 155)
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (int(new_x), int(y), size, size))

        # Road
        vanish_x, vanish_y = self.WIDTH // 2, self.HEIGHT // 2
        road_width_bottom = self.WIDTH * 1.5
        road_width_top = self.WIDTH * 0.1
        
        points = [
            ((self.WIDTH - road_width_bottom) / 2, self.HEIGHT),
            ((self.WIDTH + road_width_bottom) / 2, self.HEIGHT),
            (vanish_x + road_width_top / 2, vanish_y),
            (vanish_x - road_width_top / 2, vanish_y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ROAD)

        # Lane lines
        for i in range(1, 6):
            y_pos = i * self.HEIGHT / 6
            start_x = 0
            end_x = self.WIDTH
            
            p_y = vanish_y + (y_pos - vanish_y)
            p_x = vanish_x + (start_x - vanish_x) * ((p_y - vanish_y) / (self.HEIGHT - vanish_y)) if self.HEIGHT != vanish_y else vanish_x
            
            # Use line segments to simulate perspective
            for j in range(10):
                seg_start = j / 10
                seg_end = (j + 0.5) / 10
                
                start_point_y = vanish_y + (self.HEIGHT - vanish_y) * seg_start**2
                end_point_y = vanish_y + (self.HEIGHT - vanish_y) * seg_end**2
                
                start_point_x_left = vanish_x - (vanish_x * (start_point_y - vanish_y) / (self.HEIGHT - vanish_y))
                end_point_x_left = vanish_x - (vanish_x * (end_point_y - vanish_y) / (self.HEIGHT - vanish_y))

                start_point_x_right = self.WIDTH - start_point_x_left
                end_point_x_right = self.WIDTH - end_point_x_left

                pygame.draw.line(self.screen, self.COLOR_LANE, (start_point_x_left, start_point_y), (end_point_x_left, end_point_y), 2)
                pygame.draw.line(self.screen, self.COLOR_LANE, (start_point_x_right, start_point_y), (end_point_x_right, end_point_y), 2)

    def _render_game(self):
        # Draw hit zone and receptor feedback
        for i in range(1, 7):
            y = self.NOTE_MAP[i]['y']
            color = self.NOTE_MAP[i]['color']
            flash_alpha = int(255 * (self.last_action_feedback[i] / 10))
            if flash_alpha > 0:
                flash_surf = pygame.Surface((self.HIT_WINDOW * 2, 20), pygame.SRCALPHA)
                flash_surf.fill((*color, flash_alpha))
                self.screen.blit(flash_surf, (self.HIT_ZONE_X - self.HIT_WINDOW, y - 10))

        pygame.draw.line(self.screen, self.COLOR_WHITE, (self.HIT_ZONE_X, 0), (self.HIT_ZONE_X, self.HEIGHT), 3)

        # Draw notes
        for note in sorted(self.notes, key=lambda n: n.shape != 'rect'): # Draw shapes on top
            x, y, color = int(note.x), int(note.y), note.color
            if note.is_missed:
                color = self.GREY

            if note.shape == 'rect':
                pygame.draw.rect(self.screen, color, (x - 15, y - 8, 30, 16), border_radius=4)
                pygame.draw.rect(self.screen, self.COLOR_WHITE, (x - 15, y - 8, 30, 16), 2, border_radius=4)
            elif note.shape == 'circle':
                pygame.gfxdraw.filled_circle(self.screen, x, y, 12, color)
                pygame.gfxdraw.aacircle(self.screen, x, y, 12, self.COLOR_WHITE)
            elif note.shape == 'diamond':
                points = [(x, y - 12), (x + 12, y), (x, y + 12), (x - 12, y)]
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_WHITE)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p.life / 20))
            color_with_alpha = (*p.color, alpha)
            temp_surf = pygame.Surface((p.size, p.size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color_with_alpha, (0, 0, p.size, p.size))
            self.screen.blit(temp_surf, (int(p.x - p.size/2), int(p.y - p.size/2)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))

        # Combo
        if self.combo > 2:
            combo_text = self.font_ui.render(f"COMBO: {self.combo}", True, self.COLOR_YELLOW)
            self.screen.blit(combo_text, (self.WIDTH - combo_text.get_width() - 10, 10))

        # Health bar
        health_percent = max(0, self.health / 100.0)
        health_bar_width = 200
        health_bar_height = 15
        health_color = self.COLOR_GREEN if health_percent > 0.5 else self.COLOR_YELLOW if health_percent > 0.25 else self.COLOR_RED
        pygame.draw.rect(self.screen, self.COLOR_GREY, (10, 35, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, health_color, (10, 35, int(health_bar_width * health_percent), health_bar_height))
        
        # Accuracy
        total_processed = self.notes_hit + self.notes_missed
        accuracy = (self.notes_hit / total_processed * 100) if total_processed > 0 else 100.0
        acc_text = self.font_ui.render(f"ACC: {accuracy:.1f}%", True, self.COLOR_WHITE)
        self.screen.blit(acc_text, (self.WIDTH - acc_text.get_width() - 10, 35))
        
        # Progress Bar
        progress = self.steps / self.MAX_STEPS
        pygame.draw.rect(self.screen, self.COLOR_GREY, (0, self.HEIGHT - 5, self.WIDTH, 5))
        pygame.draw.rect(self.screen, self.COLOR_PURPLE, (0, self.HEIGHT - 5, self.WIDTH * progress, 5))

    def _get_info(self):
        total_processed = self.notes_hit + self.notes_missed
        accuracy = (self.notes_hit / total_processed) if total_processed > 0 else 1.0
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
            "combo": self.combo,
            "accuracy": accuracy,
        }

    def close(self):
        pygame.font.quit()
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
        assert not trunc
        assert isinstance(info, dict)
        
        # Test specific mechanics from brief
        self.reset()
        self.steps = 199
        self.step(self.action_space.sample())
        # note speed at step 200 is 3.0 + (200 // 200) * 0.15 = 3.15
        assert math.isclose(self.note_speed, 3.15), f"Note speed check failed. Expected 3.15, got {self.note_speed}"
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Rhythm Highway")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward}, Accuracy: {info['accuracy']:.2%}")
            obs, info = env.reset()
            total_reward = 0

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)
        
    env.close()