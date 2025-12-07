
# Generated: 2025-08-28T03:43:23.576748
# Source Brief: brief_05014.md
# Brief Index: 5014

        
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
        "Controls: Press Space to hit the notes as they enter the hit zone."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-futuristic rhythm game. Hit the notes on beat to maintain your momentum and race to the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG_TOP = (10, 5, 30)
    COLOR_BG_BOTTOM = (30, 10, 50)
    COLOR_ROAD_LINE = (100, 80, 200, 100)
    COLOR_NOTE_DEFAULT = (0, 200, 255)
    COLOR_NOTE_GLOW = (150, 220, 255)
    COLOR_HIT_ZONE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_MOMENTUM_BAR_BG = (50, 50, 80)
    COLOR_MOMENTUM_HIGH = (0, 255, 128)
    COLOR_MOMENTUM_LOW = (255, 50, 50)
    COLOR_PARTICLE_HIT = (200, 255, 255)
    COLOR_PARTICLE_MISS = (255, 100, 100)

    # Game parameters
    MAX_STEPS = 1000
    MOMENTUM_MAX = 100
    MOMENTUM_START = 75
    MOMENTUM_DECAY = 0.08
    MOMENTUM_GAIN_ON_HIT = 5
    MOMENTUM_LOSS_ON_MISS = 10
    MOMENTUM_LOSS_ON_MISSCLICK = 5
    
    NOTE_INITIAL_SPEED = 4.0
    NOTE_SPEED_INCREASE_INTERVAL = 200
    NOTE_SPEED_INCREASE_AMOUNT = 0.5
    
    HIT_ZONE_X = 120
    HIT_ZONE_WIDTH = 20
    NOTE_HIT_TOLERANCE = 25 # pixels
    
    CHECKPOINT_NOTE_COUNT = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_large = pygame.font.SysFont("Consolas", 36)
            self.font_medium = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 42)
            self.font_medium = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 22)

        self._create_background()
        
        # Initialize state variables
        self.notes = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.momentum = 0
        self.note_speed = 0
        self.prev_space_held = False
        self.hits_since_checkpoint = 0
        self.road_offset = 0.0
        self.note_hit_effect = 0
        self.note_miss_effect = 0
        self.next_note_step = 0
        
        self.reset()
        
        self.validate_implementation()

    def _create_background(self):
        self.bg_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.bg_surface, color, (0, y), (self.SCREEN_WIDTH, y))
        
        self.stars = [
            (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(1, 2))
            for _ in range(100)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.momentum = self.MOMENTUM_START
        self.note_speed = self.NOTE_INITIAL_SPEED
        self.prev_space_held = False
        self.hits_since_checkpoint = 0
        self.road_offset = 0.0
        self.note_hit_effect = 0
        self.note_miss_effect = 0

        self.notes = []
        self.particles = []
        
        self.next_note_step = self.np_random.integers(30, 60)

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            shift_held = action[2] == 1  # Boolean
            
            space_pressed = space_held and not self.prev_space_held
            self.prev_space_held = space_held

            # --- Update Game State ---
            self.steps += 1
            self.road_offset = (self.road_offset + self.note_speed) % self.SCREEN_WIDTH
            self.note_hit_effect = max(0, self.note_hit_effect - 1)
            self.note_miss_effect = max(0, self.note_miss_effect - 1)

            # Momentum decay
            self.momentum = max(0, self.momentum - self.MOMENTUM_DECAY)

            # Update notes
            for note in self.notes:
                if note['state'] == 'active':
                    note['x'] -= self.note_speed
                    # Check for missed notes
                    if note['x'] < self.HIT_ZONE_X - self.NOTE_HIT_TOLERANCE:
                        note['state'] = 'missed'
                        note['lifetime'] = 20 # frames to show miss effect
                        reward -= 1
                        self.score -= 5
                        self.momentum -= self.MOMENTUM_LOSS_ON_MISS
                        self._create_particles(note['x'], note['y'], self.COLOR_PARTICLE_MISS)
                        self.note_miss_effect = 10 # frames for screen flash
                        # sfx: miss_sound

            # Handle player action (space press)
            if space_pressed:
                hit_a_note = False
                for note in self.notes:
                    if note['state'] == 'active' and abs(note['x'] - self.HIT_ZONE_X) < self.NOTE_HIT_TOLERANCE:
                        note['state'] = 'hit'
                        note['lifetime'] = 20 # frames to show hit effect
                        reward += 1
                        self.score += 10
                        self.momentum += self.MOMENTUM_GAIN_ON_HIT
                        self._create_particles(note['x'], note['y'], self.COLOR_PARTICLE_HIT)
                        self.note_hit_effect = 10 # frames for screen flash
                        hit_a_note = True
                        self.hits_since_checkpoint += 1
                        if self.hits_since_checkpoint >= self.CHECKPOINT_NOTE_COUNT:
                            reward += 5
                            self.hits_since_checkpoint = 0
                            # sfx: checkpoint_sound
                        # sfx: hit_sound
                        break # Only hit one note per press
                
                if not hit_a_note:
                    # Miss-click penalty
                    reward -= 1
                    self.momentum -= self.MOMENTUM_LOSS_ON_MISSCLICK
                    self.note_miss_effect = 5

            self.momentum = max(0, min(self.MOMENTUM_MAX, self.momentum))
            
            # Remove old notes and effects
            self.notes = [n for n in self.notes if n['x'] > -50 and n['lifetime'] > 0]
            
            # Generate new notes
            if self.steps >= self.next_note_step:
                self.notes.append({
                    'x': self.SCREEN_WIDTH + 50,
                    'y': self.SCREEN_HEIGHT / 2,
                    'state': 'active',
                    'lifetime': 1
                })
                min_gap = int(30 / (self.note_speed / self.NOTE_INITIAL_SPEED))
                max_gap = int(70 / (self.note_speed / self.NOTE_INITIAL_SPEED))
                self.next_note_step = self.steps + self.np_random.integers(min_gap, max_gap)

            # Update particles
            for p in self.particles:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['lifetime'] -= 1
            self.particles = [p for p in self.particles if p['lifetime'] > 0]
            
            # Increase difficulty
            if self.steps > 0 and self.steps % self.NOTE_SPEED_INCREASE_INTERVAL == 0:
                self.note_speed += self.NOTE_SPEED_INCREASE_AMOUNT

            # --- Check Termination ---
            if self.momentum <= 0:
                terminated = True
                self.game_over = True
                reward -= 100
                # sfx: game_over_sound
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                self.game_over = True
                if self.momentum > 50:
                    reward += 50
                    # sfx: victory_sound
                else:
                    reward -= 50
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _create_particles(self, x, y, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'lifetime': random.randint(15, 25),
                'color': color,
                'size': random.uniform(1, 3)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.bg_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Stars ---
        for x, y, speed in self.stars:
            px = (x - self.road_offset * 0.1 * speed) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, (200, 200, 255), (int(px), y), 1)

        # --- Road ---
        horizon_y = self.SCREEN_HEIGHT * 0.4
        vanishing_point = (self.SCREEN_WIDTH / 2, horizon_y)
        
        for i in range(20):
            # Horizontal lines
            y = horizon_y + i * i * 0.8
            if y > self.SCREEN_HEIGHT: break
            
            scroll_x = (self.road_offset * (i*0.1 + 1)) % 40
            for j in range(-1, self.SCREEN_WIDTH // 40 + 1):
                start_x = j * 40 - scroll_x
                pygame.draw.line(self.screen, self.COLOR_ROAD_LINE, (start_x, y), (start_x + 20, y))

            # Perspective lines
            start_y = horizon_y
            end_y = self.SCREEN_HEIGHT
            
            x_factor = i * 0.1
            left_start_x = vanishing_point[0] - self.SCREEN_WIDTH * x_factor
            left_end_x = vanishing_point[0] - self.SCREEN_WIDTH * x_factor * 3
            
            right_start_x = vanishing_point[0] + self.SCREEN_WIDTH * x_factor
            right_end_x = vanishing_point[0] + self.SCREEN_WIDTH * x_factor * 3

            if i > 0:
                pygame.draw.aaline(self.screen, self.COLOR_ROAD_LINE, (int(left_start_x), int(start_y)), (int(left_end_x), int(end_y)))
                pygame.draw.aaline(self.screen, self.COLOR_ROAD_LINE, (int(right_start_x), int(start_y)), (int(right_end_x), int(end_y)))

        # --- Hit Zone ---
        hit_zone_rect = pygame.Rect(self.HIT_ZONE_X - self.HIT_ZONE_WIDTH / 2, 0, self.HIT_ZONE_WIDTH, self.SCREEN_HEIGHT)
        s = pygame.Surface((self.HIT_ZONE_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        
        alpha = 60
        if self.note_hit_effect > 0:
            alpha += self.note_hit_effect * 15
        elif self.note_miss_effect > 0:
            alpha += self.note_miss_effect * 10
            
        color = (*self.COLOR_HIT_ZONE, min(255, alpha))
        s.fill(color)
        self.screen.blit(s, (hit_zone_rect.x, 0))

        # --- Notes ---
        for note in self.notes:
            size_factor = 1.0 - min(1, (note['x'] - self.HIT_ZONE_X) / self.SCREEN_WIDTH)
            size = int(10 + 30 * size_factor)
            pos = (int(note['x']), int(note['y']))

            if note['state'] == 'active':
                # Draw glow
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(size * 1.2), (*self.COLOR_NOTE_GLOW, 50))
                # Draw note
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_NOTE_DEFAULT)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_NOTE_DEFAULT)

        # --- Particles ---
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 25))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'], p['size']), p['size'])
            self.screen.blit(s, (p['x']-p['size'], p['y']-p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # --- Score and Steps ---
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        steps_text = self.font_medium.render(f"TRACK: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # --- Momentum Bar ---
        bar_height = 20
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        bar_width = self.SCREEN_WIDTH - 20
        
        pygame.draw.rect(self.screen, self.COLOR_MOMENTUM_BAR_BG, (10, bar_y, bar_width, bar_height))
        
        momentum_ratio = self.momentum / self.MOMENTUM_MAX
        filled_width = int(bar_width * momentum_ratio)
        
        color = (
            self.COLOR_MOMENTUM_LOW[0] * (1 - momentum_ratio) + self.COLOR_MOMENTUM_HIGH[0] * momentum_ratio,
            self.COLOR_MOMENTUM_LOW[1] * (1 - momentum_ratio) + self.COLOR_MOMENTUM_HIGH[1] * momentum_ratio,
            self.COLOR_MOMENTUM_LOW[2] * (1 - momentum_ratio) + self.COLOR_MOMENTUM_HIGH[2] * momentum_ratio,
        )
        
        if filled_width > 0:
            pygame.draw.rect(self.screen, color, (10, bar_y, filled_width, bar_height))
        
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, bar_y, bar_width, bar_height), 2)
        
        # --- Game Over Text ---
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            message = ""
            if self.momentum <= 0:
                message = "MOMENTUM DEPLETED"
            elif self.steps >= self.MAX_STEPS:
                if self.momentum > 50:
                    message = "TRACK COMPLETE!"
                else:
                    message = "FINISH LINE CROSSED"
            
            game_over_text = self.font_large.render(message, True, self.COLOR_UI_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "momentum": self.momentum,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythm Racer")
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
        
        # The observation is already the rendered frame, just need to show it
        # Pygame uses (width, height), but numpy array is (height, width, channels)
        # Transpose back for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.FPS)
        
    env.close()