
# Generated: 2025-08-28T06:47:48.877466
# Source Brief: brief_03037.md
# Brief Index: 3037

        
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

    user_guide = (
        "Controls: Press space to slice the notes as they cross the line."
    )

    game_description = (
        "Slice incoming notes to the beat in this side-view rhythm action game."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- Game Constants ---
        self.FPS = 30 # Rate at which steps are called
        self.SLICE_LINE_X = 150
        self.MAX_MISSES = 5
        
        # Timing windows (in pixels from the slice line)
        self.PERFECT_WINDOW = 8
        self.GOOD_WINDOW = 20
        self.MISS_WINDOW = 35 # Notes passing this are considered missed

        # Colors
        self.COLOR_BG_TOP = (44, 0, 62)
        self.COLOR_BG_BOTTOM = (12, 0, 46)
        self.COLOR_SLICE_LINE = (220, 220, 255)
        self.COLOR_NOTE = (0, 255, 255)
        self.COLOR_NOTE_GLOW = (0, 150, 150)
        self.COLOR_PARTICLE_PERFECT = (100, 255, 100)
        self.COLOR_PARTICLE_GOOD = (255, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.COLOR_RED = (255, 50, 50)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.combo = 0
        self.missed_notes_count = 0
        self.game_over = False
        self.song_finished_successfully = False
        
        self.bpm = 0.0
        self.notes = []
        self.total_notes_in_song = 0
        self.notes_hit = 0
        self.song_end_step = 0

        self.particles = []
        self.slice_effect = {"timer": 0, "accuracy": None}
        self.last_space_held = False

        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.combo = 0
        self.missed_notes_count = 0
        self.game_over = False
        self.song_finished_successfully = False
        
        self.bpm = 60.0
        self.notes = self._generate_song()
        self.total_notes_in_song = len(self.notes)
        self.notes_hit = 0
        if self.notes:
            self.song_end_step = self.notes[-1]['hit_step'] + int(self.FPS * 2)
        else:
            self.song_end_step = 1

        self.particles = []
        self.slice_effect = {"timer": 0, "accuracy": None}
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()

    def _generate_song(self):
        notes = []
        current_step = int(self.FPS * 2) # Start first note after 2 seconds
        song_duration_steps = int(self.FPS * 50) # Approx 50 second song

        while current_step < song_duration_steps:
            # BPM at the time the note is generated
            current_bpm = 60.0 + 5.0 * math.floor(current_step / 60.0)
            steps_per_beat = (60.0 / current_bpm) * self.FPS

            notes.append({
                'hit_step': int(current_step),
                'state': 'upcoming', # upcoming, hit_perfect, hit_good, missed
                'y': self.np_random.integers(self.HEIGHT * 0.3, self.HEIGHT * 0.7),
                'x': self.WIDTH + 50 # Start off-screen
            })
            
            step_increment = steps_per_beat * self.np_random.choice([0.5, 1.0, 1.5, 2.0])
            current_step += max(self.FPS * 0.3, step_increment) # Ensure minimum spacing
        
        return notes

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Action Handling ---
        _ , space_held, _ = action
        is_slicing = space_held and not self.last_space_held
        self.last_space_held = space_held

        # --- Game Logic Update ---
        # 1. Update BPM
        if self.steps > 0 and self.steps % 60 == 0:
            self.bpm += 5

        # 2. Update note positions and check for misses
        note_speed = (self.WIDTH - self.SLICE_LINE_X) / ((60.0 / self.bpm) * self.FPS * 2.0)
        
        for note in self.notes:
            if note['state'] == 'upcoming':
                note['x'] -= note_speed
                if note['x'] < self.SLICE_LINE_X - self.MISS_WINDOW:
                    note['state'] = 'missed'
                    self.missed_notes_count += 1
                    self.combo = 0
                    reward -= 1 # Late slice penalty
                    self.slice_effect = {"timer": 10, "accuracy": "miss"}
                    # Sound: Miss

        # 3. Handle player slicing
        if is_slicing:
            # Sound: Slice_Swing
            self.slice_effect = {"timer": 15, "accuracy": "miss"} # Default to miss
            
            hit_a_note = False
            for note in sorted(self.notes, key=lambda n: abs(n['x'] - self.SLICE_LINE_X)):
                if note['state'] == 'upcoming':
                    dist = abs(note['x'] - self.SLICE_LINE_X)
                    if dist <= self.PERFECT_WINDOW:
                        note['state'] = 'hit_perfect'
                        self.score += 10
                        reward += 2
                        self.combo += 1
                        self.notes_hit += 1
                        self.slice_effect = {"timer": 15, "accuracy": "perfect"}
                        self._create_particles(self.SLICE_LINE_X, note['y'], self.COLOR_PARTICLE_PERFECT)
                        hit_a_note = True
                        # Sound: Hit_Perfect
                        break 
                    elif dist <= self.GOOD_WINDOW:
                        note['state'] = 'hit_good'
                        self.score += 5
                        reward += 1
                        self.combo += 1
                        self.notes_hit += 1
                        self.slice_effect = {"timer": 15, "accuracy": "good"}
                        self._create_particles(self.SLICE_LINE_X, note['y'], self.COLOR_PARTICLE_GOOD)
                        hit_a_note = True
                        # Sound: Hit_Good
                        break
            
            if not hit_a_note:
                reward -= 0.1 # Penalty for slicing at nothing
                self.combo = 0
        
        # 4. Handle combo reward
        if self.combo > 0 and self.combo % 10 == 0:
            reward += 5
            # Could add a special effect for combo milestones

        # 5. Update visual effects
        if self.slice_effect["timer"] > 0:
            self.slice_effect["timer"] -= 1
        self._update_particles()
        
        # --- Termination Check ---
        terminated = False
        if self.missed_notes_count >= self.MAX_MISSES:
            self.game_over = True
            terminated = True
            reward = -100
        
        all_notes_processed = all(n['state'] != 'upcoming' for n in self.notes)
        if self.total_notes_in_song > 0 and all_notes_processed:
            self.game_over = True
            terminated = True
            if self.missed_notes_count < self.MAX_MISSES:
                self.song_finished_successfully = True
                reward = 100
            else: # Finished song but with too many misses
                reward = -100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _create_particles(self, x, y, color):
        for _ in range(self.np_random.integers(15, 25)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 7)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [x, y], 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.98 # Friction
            p['vel'][1] *= 0.98
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        # 1. Draw background
        self._draw_gradient_background()
        
        # 2. Render game elements
        self._render_game()
        
        # 3. Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_gradient_background(self):
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp),
                int(self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp),
                int(self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw slice line and its effects
        line_color = self.COLOR_SLICE_LINE
        if self.slice_effect["timer"] > 0:
            alpha = int(255 * (self.slice_effect["timer"] / 15.0))
            if self.slice_effect["accuracy"] == "perfect":
                flash_color = self.COLOR_PARTICLE_PERFECT
            elif self.slice_effect["accuracy"] == "good":
                flash_color = self.COLOR_PARTICLE_GOOD
            else: # miss
                flash_color = self.COLOR_RED
            
            # Draw a glowing flash
            radius = int(30 * (1.0 - self.slice_effect["timer"] / 15.0))
            s = pygame.Surface((radius * 2, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.rect(s, (*flash_color, int(alpha*0.5)), (0, 0, radius*2, self.HEIGHT), border_radius=radius)
            self.screen.blit(s, (self.SLICE_LINE_X - radius, 0))

        pygame.draw.line(self.screen, line_color, (self.SLICE_LINE_X, 0), (self.SLICE_LINE_X, self.HEIGHT), 2)

        # Draw notes
        for note in self.notes:
            if note['state'] == 'upcoming':
                x, y = int(note['x']), int(note['y'])
                # Scale note size as it approaches
                dist_factor = max(0, 1 - (note['x'] - self.SLICE_LINE_X) / self.WIDTH)
                radius = int(10 + 10 * dist_factor)
                
                # Draw glow then circle
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius + 3, (*self.COLOR_NOTE_GLOW, 100))
                pygame.gfxdraw.aacircle(self.screen, x, y, radius, self.COLOR_NOTE)
                pygame.gfxdraw.filled_circle(self.screen, x, y, radius, self.COLOR_NOTE)

        # Draw particles
        for p in self.particles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color = (*p['color'], alpha)
            s = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (3, 3), 3)
            self.screen.blit(s, (x - 3, y - 3))

    def _render_ui(self):
        # Draw text with a simple shadow
        def draw_text(text, font, color, x, y, shadow=True):
            if shadow:
                text_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surface, (x + 2, y + 2))
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (x, y))

        # Score, Combo, Misses
        draw_text(f"SCORE: {self.score}", self.font_small, self.COLOR_TEXT, 10, 10)
        draw_text(f"COMBO: {self.combo}", self.font_small, self.COLOR_TEXT, 10, 35)
        
        miss_text = f"MISSES: {self.missed_notes_count}/{self.MAX_MISSES}"
        miss_color = self.COLOR_RED if self.missed_notes_count >= 3 else self.COLOR_TEXT
        draw_text(miss_text, self.font_small, miss_color, 10, 60)

        # BPM display
        draw_text(f"BPM: {self.bpm:.0f}", self.font_small, self.COLOR_TEXT, self.WIDTH - 120, 10)

        # Game Over / Cleared message
        if self.game_over:
            if self.song_finished_successfully:
                draw_text("SONG CLEARED", self.font_large, self.COLOR_PARTICLE_PERFECT, self.WIDTH // 2 - 200, self.HEIGHT // 2 - 30)
            else:
                draw_text("GAME OVER", self.font_large, self.COLOR_RED, self.WIDTH // 2 - 150, self.HEIGHT // 2 - 30)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "misses": self.missed_notes_count,
            "bpm": self.bpm,
            "notes_hit": self.notes_hit,
            "total_notes": self.total_notes_in_song,
        }

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import time

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Rhythm Slicer")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    running = True
    total_reward = 0
    
    # Action state
    action = env.action_space.sample()
    action.fill(0)

    while running:
        # --- Pygame Event Handling for Manual Play ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
            # Wait a moment before auto-resetting
            time.sleep(2)
            obs, info = env.reset()
            total_reward = 0

        # Reset the action for the next frame if it's a press-like action
        action[1] = 0 # Space is a press, not a hold for this manual loop

        # --- Rendering for Manual Play ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    pygame.quit()