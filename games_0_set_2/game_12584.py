import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:38:04.321813
# Source Brief: brief_02584.md
# Brief Index: 2584
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Synthwave Coder: A cyberpunk rhythm game Gymnasium environment.
    
    The player controls a receptor at the bottom of the screen, moving left and right
    to catch falling "code snippets" in time with a simulated beat.
    
    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up(noop), 2=down(noop), 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Activates "Data Surge" ability
    - actions[2]: Shift button (0=released, 1=held) -> No-op
    
    Observation Space: Box(0, 255, (400, 640, 3), uint8) - RGB image of the game.
    
    Reward Structure:
    - +1.0 for each correctly caught code snippet.
    - -1.0 for each missed snippet.
    - +0.5 for each snippet cleared by the "Data Surge" ability.
    - +50.0 for successfully completing the track (reaching max steps).
    - -50.0 for failing the track (10 consecutive misses).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A cyberpunk rhythm game where you catch falling code snippets in time with the beat. "
        "Use your 'Data Surge' ability to clear the screen."
    )
    user_guide = (
        "Controls: Use ←→ to move the receptor. Press space to activate 'Data Surge' and clear all snippets."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30 # Assumed FPS for smooth visual interpolation
    MAX_STEPS = 2000
    FAILURE_MISS_COUNT = 10

    # --- Colors (Synthwave Palette) ---
    COLOR_BG = (16, 0, 32) # #100020 Dark Purple
    COLOR_GRID = (50, 20, 80)
    COLOR_PLAYER = (0, 255, 255) # Cyan
    COLOR_PLAYER_GLOW = (0, 128, 128)
    COLOR_NOTE = (255, 0, 128) # Magenta
    COLOR_HIT_ZONE = (255, 255, 255, 30) # Semi-transparent white
    COLOR_TEXT = (220, 220, 255)
    COLOR_SUCCESS_PARTICLE = (0, 255, 128)
    COLOR_FAIL_PARTICLE = (255, 50, 50)
    COLOR_SURGE_PARTICLE = (255, 255, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode

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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Parameters ---
        self.receptor_width = 80
        self.receptor_speed = 15
        self.note_width = 20
        self.note_height = 40
        self.hit_zone_y = self.HEIGHT - 70
        self.hit_zone_height = 20
        self.note_spawn_interval = 20 # Steps between new notes
        self.data_surge_cooldown_max = 150 # 5 seconds at 30 FPS
        
        # --- State variables are initialized in reset() ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.receptor_x = 0
        self.notes = []
        self.particles = []
        self.note_speed_initial = 2.0
        self.note_speed = 0.0
        self.consecutive_misses = 0
        self.last_note_spawn_step = 0
        self.data_surge_cooldown = 0
        
        # This is called in the original code, but it's not standard Gym API.
        # We'll keep it as it was, but note that it's not necessary for a valid env.
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.receptor_x = self.WIDTH / 2
        self.notes = []
        self.particles = []
        
        self.note_speed = self.note_speed_initial
        self.consecutive_misses = 0
        self.last_note_spawn_step = 0
        self.data_surge_cooldown = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- 1. Handle Input & Player Actions ---
        reward += self._handle_input(action)

        # --- 2. Update Game Logic ---
        miss_penalty = self._update_game_state()
        reward += miss_penalty

        # --- 3. Collision Detection & Scoring ---
        hit_reward = self._check_collisions()
        reward += hit_reward
        
        # --- 4. Check for Termination ---
        terminated = False
        if self.consecutive_misses >= self.FAILURE_MISS_COUNT:
            # Failure condition
            reward -= 50
            terminated = True
            self.game_over = True
            # sfx: track_fail_sound
        elif self.steps >= self.MAX_STEPS:
            # Success condition
            reward += 50
            terminated = True
            self.game_over = True
            # sfx: track_complete_sound
        
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        reward = 0
        
        # Action 0: Movement
        if movement == 3: # Left
            self.receptor_x -= self.receptor_speed
        elif movement == 4: # Right
            self.receptor_x += self.receptor_speed
        self.receptor_x = np.clip(self.receptor_x, self.receptor_width / 2, self.WIDTH - self.receptor_width / 2)

        # Action 1: Space Bar (Data Surge)
        if space_held and self.data_surge_cooldown == 0:
            # sfx: data_surge_activate
            self.data_surge_cooldown = self.data_surge_cooldown_max
            cleared_count = len(self.notes)
            reward += 0.5 * cleared_count # Event-based reward
            for note in self.notes:
                self._spawn_particles(
                    (note['x'], note['y']), 30, self.COLOR_SURGE_PARTICLE, 4
                )
            self.notes.clear()
        
        return reward

    def _update_game_state(self):
        miss_penalty = 0

        # Update cooldowns
        if self.data_surge_cooldown > 0:
            self.data_surge_cooldown -= 1
        
        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.note_speed += 0.25 # Increased for more noticeable effect

        # Spawn new notes
        if self.steps - self.last_note_spawn_step >= self.note_spawn_interval:
            self._spawn_note()
            self.last_note_spawn_step = self.steps

        # Update notes
        notes_to_remove = []
        for note in self.notes:
            note['y'] += self.note_speed
            if note['y'] > self.HEIGHT:
                notes_to_remove.append(note)
                miss_penalty -= 1.0 # Continuous feedback reward
                self.consecutive_misses += 1
                self.score -= 1 # Score penalty for miss
                # sfx: note_miss_sound
                self._spawn_particles(
                    (note['x'], self.hit_zone_y + self.hit_zone_height / 2), 
                    15, self.COLOR_FAIL_PARTICLE, 2
                )
        self.notes = [n for n in self.notes if n not in notes_to_remove]

        # Update particles
        particles_to_remove = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

        return miss_penalty

    def _check_collisions(self):
        hit_reward = 0
        notes_to_remove = []
        hit_zone_rect = pygame.Rect(0, self.hit_zone_y, self.WIDTH, self.hit_zone_height)

        for note in self.notes:
            note_rect = pygame.Rect(note['x'] - self.note_width/2, note['y'] - self.note_height/2, self.note_width, self.note_height)
            if note_rect.colliderect(hit_zone_rect):
                receptor_hit_box = pygame.Rect(self.receptor_x - self.receptor_width/2, self.hit_zone_y, self.receptor_width, self.hit_zone_height)
                if note_rect.colliderect(receptor_hit_box):
                    notes_to_remove.append(note)
                    hit_reward += 1.0 # Continuous feedback reward
                    self.score += 10 # Game score
                    self.consecutive_misses = 0
                    # sfx: note_hit_sound
                    self._spawn_particles(
                        (note['x'], note['y']), 50, self.COLOR_SUCCESS_PARTICLE, 3
                    )
        
        if notes_to_remove:
            self.notes = [n for n in self.notes if n not in notes_to_remove]
        
        return hit_reward

    def _spawn_note(self):
        x_pos = self.np_random.uniform(self.note_width, self.WIDTH - self.note_width)
        self.notes.append({'x': x_pos, 'y': -self.note_height})

    def _spawn_particles(self, pos, count, color, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, speed_scale)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    def _get_observation(self):
        # --- 1. Fill Background & Grid ---
        self.screen.fill(self.COLOR_BG)
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

        # --- 2. Render Hit Zone ---
        hit_zone_surface = pygame.Surface((self.WIDTH, self.hit_zone_height), pygame.SRCALPHA)
        hit_zone_surface.fill(self.COLOR_HIT_ZONE)
        self.screen.blit(hit_zone_surface, (0, self.hit_zone_y))
        
        # --- 3. Render Particles ---
        for p in self.particles:
            life_ratio = p['life'] / 30.0
            radius = int(max(1, 3 * life_ratio))
            # Fast drawing for many particles
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), radius)

        # --- 4. Render Notes ---
        for note in self.notes:
            note_rect = pygame.Rect(note['x'] - self.note_width/2, note['y'] - self.note_height/2, self.note_width, self.note_height)
            pygame.draw.rect(self.screen, self.COLOR_NOTE, note_rect, border_radius=3)
            # Add a white core for better visibility
            core_rect = note_rect.inflate(-8, -8)
            pygame.draw.rect(self.screen, (255,255,255), core_rect, border_radius=2)

        # --- 5. Render Receptor ---
        receptor_pos_int = (int(self.receptor_x), int(self.hit_zone_y + self.hit_zone_height / 2))
        # Glow effect
        glow_radius = int(self.receptor_width / 2 * 1.5)
        # Using gfxdraw for anti-aliasing
        pygame.gfxdraw.filled_circle(self.screen, receptor_pos_int[0], receptor_pos_int[1], glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, receptor_pos_int[0], receptor_pos_int[1], glow_radius, self.COLOR_PLAYER_GLOW)
        # Main receptor bar
        receptor_rect = pygame.Rect(0, 0, self.receptor_width, 10)
        receptor_rect.center = receptor_pos_int
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, receptor_rect, border_radius=5)
        
        # --- 6. Render UI ---
        self._render_ui()

        # --- 7. Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score Display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Misses Display
        miss_text = self.font_ui.render(f"MISS STREAK: {self.consecutive_misses}/{self.FAILURE_MISS_COUNT}", True, self.COLOR_TEXT)
        miss_color = self.COLOR_FAIL_PARTICLE if self.consecutive_misses > 0 else self.COLOR_TEXT
        miss_text = self.font_ui.render(f"MISS STREAK: {self.consecutive_misses}/{self.FAILURE_MISS_COUNT}", True, miss_color)
        self.screen.blit(miss_text, (self.WIDTH - miss_text.get_width() - 10, 10))

        # Data Surge Cooldown
        if self.data_surge_cooldown > 0:
            cooldown_ratio = self.data_surge_cooldown / self.data_surge_cooldown_max
            bar_width = 100
            bar_height = 10
            fill_width = int(bar_width * cooldown_ratio)
            pygame.draw.rect(self.screen, self.COLOR_GRID, (self.WIDTH/2 - bar_width/2, 15, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_SURGE_PARTICLE, (self.WIDTH/2 - bar_width/2, 15, fill_width, bar_height))
        else:
            surge_text = self.font_ui.render("DATA SURGE READY", True, self.COLOR_SURGE_PARTICLE)
            self.screen.blit(surge_text, (self.WIDTH/2 - surge_text.get_width()/2, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.consecutive_misses >= self.FAILURE_MISS_COUNT:
                msg = "TRACK FAILED"
                color = self.COLOR_FAIL_PARTICLE
            else:
                msg = "TRACK COMPLETE"
                color = self.COLOR_SUCCESS_PARTICLE
            
            end_text = self.font_msg.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2 - 20))
            final_score_text = self.font_ui.render(f"FINAL SCORE: {self.score}", True, self.COLOR_TEXT)
            self.screen.blit(final_score_text, (self.WIDTH/2 - final_score_text.get_width()/2, self.HEIGHT/2 + 20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "consecutive_misses": self.consecutive_misses,
            "note_speed": self.note_speed,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
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
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to run the file directly to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Synthwave Coder")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()