import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:50:04.784403
# Source Brief: brief_02965.md
# Brief Index: 2965
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a bouncing ball to hit moving targets.
    
    The goal is to reach a score of 100 by hitting targets. The player has 3 lives.
    A life is lost if the ball is launched and comes to a stop without hitting any target.
    
    Actions:
    - Movement (up/down/left/right): Adjusts the launch angle and power when the ball is ready.
    - Spacebar: Launches the ball.
    - Shift: Activates a speed boost on launch, if available.
    
    Visuals:
    - Clean, vibrant 2D graphics with particle effects and glows.
    - Clear UI for score, lives, and boost status.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Control a bouncing ball to hit moving targets and score points. "
        "Use precise aiming and power to clear the screen and achieve a high score."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim and ↑↓ to adjust power. "
        "Press space to launch the ball. Hold shift while launching to use a speed boost when available."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.WIN_SCORE = 100
        self.MAX_STEPS = 1000
        self.INITIAL_LIVES = 3
        self.NUM_TARGETS = 5
        self.COMBO_FOR_BOOST = 3 # Reduced from 5 for more frequent use

        # Physics
        self.GRAVITY = 0.2
        self.FRICTION = 0.995
        self.MIN_SPEED_THRESHOLD = 0.1
        self.BALL_RADIUS = 12
        self.TARGET_MIN_RADIUS = 8
        self.TARGET_MAX_RADIUS = 16
        self.LAUNCH_POWER_MIN = 5.0
        self.LAUNCH_POWER_MAX = 15.0
        self.LAUNCH_POWER_SENSITIVITY = 0.2
        self.LAUNCH_ANGLE_SENSITIVITY = 0.05
        self.BOOST_MULTIPLIER = 1.5

        # Colors
        self.COLOR_BG_TOP = (10, 20, 40)
        self.COLOR_BG_BOTTOM = (0, 0, 10)
        self.COLOR_BALL = (0, 255, 128)
        self.COLOR_BALL_GLOW = (150, 255, 200)
        self.COLOR_BOOST_AURA = (255, 255, 255)
        self.COLOR_AIMER = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TARGET_LOW = (255, 80, 80) # Red
        self.COLOR_TARGET_MID = (80, 150, 255) # Blue
        self.COLOR_TARGET_HIGH = (255, 215, 0) # Gold

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
            self.game_font_small = pygame.font.SysFont("Consolas", 20, bold=True)
            self.game_font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        except pygame.error:
            self.game_font_small = pygame.font.Font(None, 24)
            self.game_font_large = pygame.font.Font(None, 36)
        
        # --- Game State Variables ---
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.ball_pos = np.array([0.0, 0.0])
        self.ball_vel = np.array([0.0, 0.0])
        self.ball_state = "ready" # "ready", "moving"
        self.launch_angle = 0.0
        self.launch_power = 0.0
        self.targets = []
        self.particles = []
        self.target_base_speed = 0.0
        self.combo_counter = 0
        self.speed_boost_available = False
        self.hit_target_this_launch = False
        
        # This is here to ensure the env is fully initialized before the first reset call
        # which might happen externally.
        # self.reset() # This is called by the wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        self.ball_state = "ready"
        self.ball_pos = np.array([self.WIDTH / 2, self.HEIGHT - self.BALL_RADIUS - 1.0])
        self.ball_vel = np.array([0.0, 0.0])
        
        self.launch_angle = -math.pi / 2  # Straight up
        self.launch_power = (self.LAUNCH_POWER_MIN + self.LAUNCH_POWER_MAX) / 2
        
        self.targets = []
        for _ in range(self.NUM_TARGETS):
            self._spawn_target()
            
        self.particles = []
        self.target_base_speed = 1.0
        self.combo_counter = 0
        self.speed_boost_available = False
        self.hit_target_this_launch = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # 1. Unpack and handle actions
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward += self._handle_input(movement, space_held, shift_held)
        
        # 2. Update game logic
        self._update_ball()
        self._update_targets()
        self._update_particles()
        
        # 3. Handle collisions and check for life loss
        points_reward = self._handle_collisions()
        reward += points_reward
        
        if self.ball_state == "stopped":
            life_loss_reward = self._check_launch_outcome()
            reward += life_loss_reward
            self._reset_ball_for_launch()
        
        # 4. Update step counter and difficulty
        self.steps += 1
        if self.steps > 0 and self.steps % 200 == 0:
            self.target_base_speed += 0.05
        
        # 5. Check for termination
        terminated = self._check_termination()
        truncated = False # No truncation condition in this game
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0  # Goal-oriented win reward
            else: # Lost all lives or timed out
                reward -= 100.0  # Goal-oriented loss reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        if self.ball_state == "ready":
            # Adjust aim
            if movement == 1: # Up
                self.launch_power = min(self.LAUNCH_POWER_MAX, self.launch_power + self.LAUNCH_POWER_SENSITIVITY)
            elif movement == 2: # Down
                self.launch_power = max(self.LAUNCH_POWER_MIN, self.launch_power - self.LAUNCH_POWER_SENSITIVITY)
            elif movement == 3: # Left
                self.launch_angle -= self.LAUNCH_ANGLE_SENSITIVITY
            elif movement == 4: # Right
                self.launch_angle += self.LAUNCH_ANGLE_SENSITIVITY
            
            # Clamp angle to upward-facing hemisphere
            self.launch_angle = max(-math.pi, min(0, self.launch_angle))
            
            # Launch
            if space_held:
                # SFX: BallLaunch.wav
                self.ball_state = "moving"
                self.hit_target_this_launch = False
                power = self.launch_power
                
                use_boost = shift_held and self.speed_boost_available
                if use_boost:
                    # SFX: BoostLaunch.wav
                    power *= self.BOOST_MULTIPLIER
                    self.speed_boost_available = False
                    self.combo_counter = 0 # Reset combo after using boost
                    return 1.0 # Reward for using boost
                
                self.ball_vel = np.array([
                    power * math.cos(self.launch_angle),
                    power * math.sin(self.launch_angle)
                ])
        return 0.0

    def _update_ball(self):
        if self.ball_state == "moving":
            self.ball_vel[1] += self.GRAVITY
            self.ball_vel *= self.FRICTION
            self.ball_pos += self.ball_vel
            
            speed = np.linalg.norm(self.ball_vel)
            if speed < self.MIN_SPEED_THRESHOLD:
                self.ball_state = "stopped"

    def _update_targets(self):
        for target in self.targets:
            target['pos'] += target['vel']
            # Wall bounces for targets
            if target['pos'][0] <= target['radius'] or target['pos'][0] >= self.WIDTH - target['radius']:
                target['vel'][0] *= -1
            if target['pos'][1] <= target['radius'] or target['pos'][1] >= self.HEIGHT - target['radius']:
                target['vel'][1] *= -1
            target['pos'][0] = np.clip(target['pos'][0], target['radius'], self.WIDTH - target['radius'])
            target['pos'][1] = np.clip(target['pos'][1], target['radius'], self.HEIGHT - target['radius'])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0.0
        # Ball vs Walls
        if self.ball_pos[0] <= self.BALL_RADIUS:
            self.ball_pos[0] = self.BALL_RADIUS
            self.ball_vel[0] *= -0.9 # Dampen bounce
            # SFX: WallBounce.wav
        elif self.ball_pos[0] >= self.WIDTH - self.BALL_RADIUS:
            self.ball_pos[0] = self.WIDTH - self.BALL_RADIUS
            self.ball_vel[0] *= -0.9
            # SFX: WallBounce.wav
        if self.ball_pos[1] <= self.BALL_RADIUS:
            self.ball_pos[1] = self.BALL_RADIUS
            self.ball_vel[1] *= -0.9
            # SFX: WallBounce.wav
        elif self.ball_pos[1] >= self.HEIGHT - self.BALL_RADIUS:
            self.ball_pos[1] = self.HEIGHT - self.BALL_RADIUS
            self.ball_vel[1] *= -0.9
            # SFX: WallBounce.wav

        # Ball vs Targets
        for i, target in reversed(list(enumerate(self.targets))):
            dist = np.linalg.norm(self.ball_pos - target['pos'])
            if dist < self.BALL_RADIUS + target['radius']:
                # SFX: TargetHit.wav
                self.score += target['points']
                reward += 0.1 # Continuous reward for any hit
                self.hit_target_this_launch = True
                self._create_particles(target['pos'], target['color'])
                self.targets.pop(i)
                self._spawn_target()
        return reward

    def _check_launch_outcome(self):
        if not self.hit_target_this_launch:
            self.lives -= 1
            self.combo_counter = 0
            self.speed_boost_available = False # Lose boost on a miss
            # SFX: LifeLost.wav
            return -5.0 # Penalty for losing a life
        else:
            self.combo_counter += 1
            if not self.speed_boost_available and self.combo_counter >= self.COMBO_FOR_BOOST:
                self.speed_boost_available = True
                # SFX: BoostReady.wav
        return 0.0

    def _reset_ball_for_launch(self):
        self.ball_state = "ready"
        self.ball_pos = np.array([self.WIDTH / 2, self.HEIGHT - self.BALL_RADIUS - 1.0])
        self.ball_vel = np.array([0.0, 0.0])

    def _check_termination(self):
        return self.score >= self.WIN_SCORE or self.lives <= 0 or self.steps >= self.MAX_STEPS

    def _spawn_target(self):
        points = self.np_random.integers(1, 11)
        radius = int(self.TARGET_MIN_RADIUS + (points / 10.0) * (self.TARGET_MAX_RADIUS - self.TARGET_MIN_RADIUS))
        
        if points <= 3: color = self.COLOR_TARGET_LOW
        elif points <= 7: color = self.COLOR_TARGET_MID
        else: color = self.COLOR_TARGET_HIGH
        
        pos = np.array([
            self.np_random.uniform(radius, self.WIDTH - radius),
            self.np_random.uniform(radius, self.HEIGHT * 0.7) # Spawn in top 70%
        ])
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.target_base_speed + self.np_random.uniform(-0.5, 0.5)
        vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
        
        self.targets.append({'pos': pos, 'vel': vel, 'radius': radius, 'points': points, 'color': color})

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': 20, 'color': color})

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio),
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game_elements(self):
        # Targets
        for target in self.targets:
            pos_int = (int(target['pos'][0]), int(target['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], target['radius'], target['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], target['radius'], target['color'])

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*p['color'], alpha)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Aiming indicator
        if self.ball_state == "ready":
            power_ratio = (self.launch_power - self.LAUNCH_POWER_MIN) / (self.LAUNCH_POWER_MAX - self.LAUNCH_POWER_MIN)
            length = 30 + 70 * power_ratio
            end_pos = (
                self.ball_pos[0] + length * math.cos(self.launch_angle),
                self.ball_pos[1] + length * math.sin(self.launch_angle)
            )
            pygame.draw.aaline(self.screen, self.COLOR_AIMER, self.ball_pos, end_pos, 2)

        # Ball
        pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        # Glow effect
        glow_radius = int(self.BALL_RADIUS * 1.8)
        if self.speed_boost_available:
            pulse = abs(math.sin(self.steps * 0.3))
            aura_radius = int(self.BALL_RADIUS * (2.0 + pulse * 0.5))
            self._render_glow(self.screen, self.COLOR_BOOST_AURA, pos_int, aura_radius, 0.2)
        else:
            self._render_glow(self.screen, self.COLOR_BALL_GLOW, pos_int, glow_radius, 0.3)
        # Ball itself
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BALL_RADIUS, self.COLOR_BALL)

    def _render_glow(self, surface, color, center, max_radius, intensity_ratio):
        for r in range(max_radius, 0, -2):
            alpha = int(255 * (1 - r / max_radius)**2 * intensity_ratio)
            if alpha > 0:
                pygame.gfxdraw.aacircle(surface, center[0], center[1], r, (*color, alpha))

    def _render_ui(self):
        # Score
        score_text = self.game_font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.game_font_large.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Boost status
        if self.speed_boost_available:
            boost_text = self.game_font_small.render("BOOST READY!", True, self.COLOR_BOOST_AURA)
            self.screen.blit(boost_text, (self.WIDTH // 2 - boost_text.get_width() // 2, 10))
        elif self.combo_counter > 0:
            combo_text = self.game_font_small.render(f"COMBO: {self.combo_counter}", True, self.COLOR_TEXT)
            self.screen.blit(combo_text, (self.WIDTH // 2 - combo_text.get_width() // 2, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "combo": self.combo_counter,
            "boost_ready": self.speed_boost_available
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you need to unset the dummy video driver:
    # del os.environ['SDL_VIDEODRIVER']
    
    # Check if we can run with display
    can_render = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"
    
    if not can_render:
        print("Cannot run in interactive mode with SDL_VIDEODRIVER='dummy'.")
        print("Exiting. To run interactively, run this script in a graphical environment.")
        exit()

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bouncing Ball Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0.0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
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
        
        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0.0
            
        clock.tick(env.FPS)
        
    env.close()