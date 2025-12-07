
# Generated: 2025-08-27T17:04:26.203718
# Source Brief: brief_01418.md
# Brief Index: 1418

        
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
    """
    An expert-designed Gymnasium environment for a minimalist archery game.

    This environment simulates a side-view archery range where the player's goal
    is to score as many points as possible with three arrows. The game features
    physics-based arrow trajectory, variable wind conditions, and a clean,
    high-quality visual presentation. The action space is designed to allow for
    nuanced control over aiming and power, promoting skill-based gameplay.

    **Game Phases:**
    1.  **AIMING**: The player adjusts the launch angle using UP/DOWN. Holding SPACE
        charges the shot power. Releasing SPACE fires the arrow.
    2.  **FIRING**: The arrow is in flight, its trajectory calculated each step based
        on initial velocity, gravity, and wind. The agent must send no-op actions
        to observe the flight.
    3.  **RESULT**: The arrow has landed. The score is displayed. Any action
        proceeds to the next shot or ends the game.

    **Scoring:**
    - Bullseye (Red): 100 points
    - Inner Ring (Yellow): 50 points
    - Outer Ring (Blue): 20 points
    - Miss: 0 points
    - A small penalty is applied for each step spent aiming to encourage quick shots.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ↑↓ to aim. Hold [SPACE] to charge power, release to fire. [SHIFT] resets aim."
    )

    # User-facing game description
    game_description = (
        "Hit the bullseye in this minimalist, side-view archery range simulator. "
        "Master angle and power against the wind to get the highest score with 3 arrows."
    )

    # Frame advance behavior
    auto_advance = False

    # --- Constants ---
    # Screen dimensions
    WIDTH, HEIGHT = 640, 400

    # Colors
    COLOR_BG = (15, 25, 40)
    COLOR_GROUND = (60, 45, 40)
    COLOR_TEXT = (220, 220, 230)
    COLOR_TEXT_DIM = (150, 150, 160)
    COLOR_SCORE = (255, 215, 0)
    COLOR_ARROW = (255, 255, 255)
    COLOR_BOW = (200, 200, 210)
    COLOR_TRAJECTORY = (255, 255, 255, 100)
    COLOR_POWER_BG = (50, 50, 60)
    COLOR_POWER_FILL = (100, 255, 100)
    
    TARGET_COLORS = {
        "bullseye": (255, 80, 80),
        "inner": (255, 220, 90),
        "outer": (100, 150, 255),
    }

    # Game parameters
    MAX_ARROWS = 3
    MAX_STEPS = 1000
    GROUND_Y = 360
    LAUNCH_X = 60
    TARGET_X = 550
    TARGET_Y = 250
    
    # Physics
    GRAVITY = 0.2
    MAX_POWER = 20.0
    MIN_ANGLE, MAX_ANGLE = -80.0, -10.0
    WIND_STRENGTH_FACTOR = 0.01

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 72)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.total_score = 0
        self.arrows_shot = 0
        self.game_over = False
        self.phase = 'AIMING'
        self.angle = 0.0
        self.power = 0.0
        self.wind = (0, 0)
        self.arrow = None
        self.particles = []
        self.stuck_arrows = []
        self.scores = []
        self.space_was_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_score = 0
        self.arrows_shot = 0
        self.scores = []
        self.game_over = False
        self.stuck_arrows = []
        self.particles = []
        
        self._prepare_next_shot()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        self.steps += 1
        
        if self.phase == 'AIMING':
            reward = self._handle_aiming_phase(movement, space_held, shift_held)
        elif self.phase == 'FIRING':
            reward = self._handle_firing_phase()
        elif self.phase == 'RESULT':
            # Any action moves to the next state
            if np.any(action != self.action_space.sample()*0): # A non-zero action
                 if self.arrows_shot >= self.MAX_ARROWS:
                    self.game_over = True
                 else:
                    self._prepare_next_shot()

        self._update_particles()

        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if terminated and self.total_score == 300: # Perfect score bonus
             reward += 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _prepare_next_shot(self):
        self.phase = 'AIMING'
        self.angle = -45.0
        self.power = 0.0
        self.arrow = None
        self.space_was_held = False
        
        # Generate new wind for the shot
        wind_speed = self.np_random.uniform(0, 5)
        wind_direction = self.np_random.choice([-1, 1])
        self.wind = (wind_speed, wind_direction)

    def _handle_aiming_phase(self, movement, space_held, shift_held):
        reward = -0.01  # Small penalty for taking time to aim

        if shift_held:
            self.angle = -45.0
            self.power = 0.0
            # sfx: reset_aim
            return reward

        if movement == 1:  # Up
            self.angle = max(self.MIN_ANGLE, self.angle - 0.5)
        elif movement == 2:  # Down
            self.angle = min(self.MAX_ANGLE, self.angle + 0.5)

        if space_held:
            self.power = min(self.MAX_POWER, self.power + 0.25)
            self.space_was_held = True
            if self.power >= self.MAX_POWER:
                # sfx: power_max_tick
                pass
        elif self.space_was_held: # Space was just released
            self._fire_arrow()
            # sfx: arrow_release
            
        return reward

    def _fire_arrow(self):
        self.phase = 'FIRING'
        rad_angle = math.radians(self.angle)
        vel_x = self.power * math.cos(rad_angle)
        vel_y = self.power * math.sin(rad_angle)
        
        self.arrow = {
            'pos': [self.LAUNCH_X, self.GROUND_Y - 50],
            'vel': [vel_x, vel_y],
            'angle': self.angle,
            'path': [[self.LAUNCH_X, self.GROUND_Y - 50]]
        }

    def _handle_firing_phase(self):
        if not self.arrow: return 0

        # Update physics
        self.arrow['vel'][1] += self.GRAVITY
        self.arrow['vel'][0] += self.wind[0] * self.wind[1] * self.WIND_STRENGTH_FACTOR
        self.arrow['pos'][0] += self.arrow['vel'][0]
        self.arrow['pos'][1] += self.arrow['vel'][1]
        self.arrow['angle'] = math.degrees(math.atan2(self.arrow['vel'][1], self.arrow['vel'][0]))
        self.arrow['path'].append(list(self.arrow['pos']))
        if len(self.arrow['path']) > 20:
             self.arrow['path'].pop(0)

        # Check for collision
        pos_x, pos_y = self.arrow['pos']
        
        # Hit target?
        if abs(pos_x - self.TARGET_X) < 5:
            dist_from_center = math.hypot(pos_x - self.TARGET_X, pos_y - self.TARGET_Y)
            if dist_from_center < 60: # Hit within the largest ring
                score = 0
                if dist_from_center < 15: # Bullseye
                    score = 100
                    # sfx: hit_bullseye
                elif dist_from_center < 35: # Inner ring
                    score = 50
                    # sfx: hit_target_good
                else: # Outer ring
                    score = 20
                    # sfx: hit_target_ok
                
                self.stuck_arrows.append({'pos': self.arrow['pos'][:], 'angle': self.arrow['angle']})
                self._create_impact_particles(self.arrow['pos'])
                return self._finalize_shot(score)

        # Hit ground or went off-screen?
        if pos_y > self.GROUND_Y or pos_x > self.WIDTH or pos_x < 0:
            # sfx: arrow_miss
            return self._finalize_shot(0)
            
        return 0

    def _finalize_shot(self, score):
        self.total_score += score
        self.scores.append(score)
        self.arrows_shot += 1
        self.phase = 'RESULT'
        self.arrow = None
        # Reward is proportional to score, scaled to be within a reasonable range for RL
        return score / 10.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        pygame.draw.line(self.screen, (10,5,5), (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)

        # Draw target
        target_radii = {"outer": 60, "inner": 35, "bullseye": 15}
        pygame.gfxdraw.filled_circle(self.screen, self.TARGET_X, self.TARGET_Y, target_radii["outer"], self.TARGET_COLORS["outer"])
        pygame.gfxdraw.filled_circle(self.screen, self.TARGET_X, self.TARGET_Y, target_radii["inner"], self.TARGET_COLORS["inner"])
        pygame.gfxdraw.filled_circle(self.screen, self.TARGET_X, self.TARGET_Y, target_radii["bullseye"], self.TARGET_COLORS["bullseye"])
        
        # Draw previous arrows
        for arr in self.stuck_arrows:
            self._draw_arrow(arr['pos'], arr['angle'])
            
        # Draw current arrow in flight (with trail)
        if self.phase == 'FIRING' and self.arrow:
            if len(self.arrow['path']) > 1:
                # Use a decreasing alpha for the trail
                for i in range(len(self.arrow['path']) - 1):
                    alpha = int(255 * (i / len(self.arrow['path'])))
                    color = (*self.COLOR_ARROW, alpha)
                    pygame.draw.line(self.screen, color, self.arrow['path'][i], self.arrow['path'][i+1], 2)
            self._draw_arrow(self.arrow['pos'], self.arrow['angle'])

        # Draw bow and trajectory preview in aiming phase
        if self.phase == 'AIMING':
            self._draw_bow()
            self._draw_trajectory_preview()
            
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['radius']))

    def _draw_bow(self):
        bow_pos = (self.LAUNCH_X, self.GROUND_Y - 50)
        # Simple arc for the bow
        rect = pygame.Rect(bow_pos[0] - 10, bow_pos[1] - 25, 20, 50)
        pygame.draw.arc(self.screen, self.COLOR_BOW, rect, math.radians(80), math.radians(280), 3)
        # Draw arrow nocked on the bow
        rad_angle = math.radians(self.angle)
        end_x = bow_pos[0] + 20 * math.cos(rad_angle)
        end_y = bow_pos[1] + 20 * math.sin(rad_angle)
        pygame.draw.line(self.screen, self.COLOR_ARROW, bow_pos, (end_x, end_y), 2)

    def _draw_arrow(self, pos, angle_deg):
        length = 30
        rad_angle = math.radians(angle_deg)
        start_pos = (pos[0] - length / 2 * math.cos(rad_angle), pos[1] - length / 2 * math.sin(rad_angle))
        end_pos = (pos[0] + length / 2 * math.cos(rad_angle), pos[1] + length / 2 * math.sin(rad_angle))
        pygame.draw.line(self.screen, self.COLOR_ARROW, start_pos, end_pos, 2)
        # Fletching
        fletch_len = 8
        fletch_angle = math.radians(150)
        fletch1_end = (start_pos[0] + fletch_len * math.cos(rad_angle + fletch_angle), start_pos[1] + fletch_len * math.sin(rad_angle + fletch_angle))
        fletch2_end = (start_pos[0] + fletch_len * math.cos(rad_angle - fletch_angle), start_pos[1] + fletch_len * math.sin(rad_angle - fletch_angle))
        pygame.draw.line(self.screen, self.COLOR_ARROW, start_pos, fletch1_end, 2)
        pygame.draw.line(self.screen, self.COLOR_ARROW, start_pos, fletch2_end, 2)

    def _draw_trajectory_preview(self):
        if self.power > 0.5:
            rad_angle = math.radians(self.angle)
            vel_x = self.power * math.cos(rad_angle)
            vel_y = self.power * math.sin(rad_angle)
            pos = [self.LAUNCH_X, self.GROUND_Y - 50]
            
            for i in range(30): # Draw 30 segments of the predicted path
                vel_y += self.GRAVITY
                vel_x += self.wind[0] * self.wind[1] * self.WIND_STRENGTH_FACTOR
                next_pos = [pos[0] + vel_x, pos[1] + vel_y]
                if i % 3 == 0: # Draw a dot every 3 steps
                     pygame.draw.circle(self.screen, self.COLOR_TRAJECTORY, (int(pos[0]), int(pos[1])), 1)
                pos = next_pos
                if pos[1] > self.GROUND_Y or pos[0] > self.WIDTH:
                    break

    def _render_ui(self):
        # Draw Power Bar
        if self.phase == 'AIMING' and self.space_was_held:
            bar_w, bar_h = 100, 20
            bar_x, bar_y = self.LAUNCH_X - bar_w / 2, self.GROUND_Y - 110
            fill_w = (self.power / self.MAX_POWER) * bar_w
            pygame.draw.rect(self.screen, self.COLOR_POWER_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_POWER_FILL, (bar_x, bar_y, fill_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

        # Draw Wind Indicator
        wind_speed, wind_dir = self.wind
        wind_text = f"WIND: {wind_speed:.1f} m/s"
        arrow_char = "→" if wind_dir > 0 else "←"
        self._draw_text(f"{arrow_char} {wind_text}", (10, 10), self.font_small, self.COLOR_TEXT_DIM)

        # Draw Score
        self._draw_text(f"SCORE: {self.total_score}", (self.WIDTH - 150, 10), self.font_main, self.COLOR_SCORE)
        
        # Draw individual arrow scores
        for i in range(self.MAX_ARROWS):
            score_str = f"A{i+1}: --"
            if i < len(self.scores):
                score_str = f"A{i+1}: {self.scores[i]}"
            self._draw_text(score_str, (self.WIDTH - 150, 45 + i * 25), self.font_small, self.COLOR_TEXT)

        # Draw Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            self._draw_text("GAME OVER", (self.WIDTH / 2, self.HEIGHT / 2 - 40), self.font_title, self.COLOR_SCORE, center=True)
            self._draw_text(f"Final Score: {self.total_score}", (self.WIDTH / 2, self.HEIGHT / 2 + 20), self.font_main, self.COLOR_TEXT, center=True)

    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_impact_particles(self, pos):
        # sfx: particle_burst
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            radius = self.np_random.uniform(2, 5)
            color = random.choice(list(self.TARGET_COLORS.values()))
            self.particles.append({'pos': list(pos), 'vel': vel, 'radius': radius, 'life': 20, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['radius'] -= 0.1
            p['life'] -= 1
            if p['radius'] <= 0 or p['life'] <= 0:
                self.particles.remove(p)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps,
            "arrows_shot": self.arrows_shot,
            "phase": self.phase,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Archery Range")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        if terminated:
            # In manual play, allow reset after game over
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    terminated = False
                    
        else: # Game is active
            # --- Action mapping for human play ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment stepping ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Event handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS for smooth human experience

    env.close()