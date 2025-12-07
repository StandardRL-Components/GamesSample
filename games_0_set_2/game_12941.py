import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a bouncing ball to collect gems within a shrinking room. Avoid getting crushed by the walls as they close in."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move horizontally and ↑↓ to apply vertical thrust. Collect gems to increase your score and bounce height."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_FPS = 30
    
    # Colors
    COLOR_BG = (20, 10, 40) # Dark Purple
    COLOR_BALL = (0, 255, 255) # Bright Cyan
    COLOR_BALL_GLOW = (0, 128, 128)
    COLOR_GEM = (255, 255, 0) # Yellow
    COLOR_GEM_GLOW = (128, 128, 0)
    COLOR_WALL_BASE = (255, 20, 147) # Deep Pink
    COLOR_TEXT = (240, 240, 240)
    
    # Physics & Gameplay
    BALL_RADIUS = 12
    GEM_RADIUS = 8
    PLAYER_HORIZONTAL_SPEED = 6.0
    PLAYER_VERTICAL_THRUST = 1.5
    GRAVITY = 0.6
    INITIAL_BOUNCE_VEL = -12.0
    WALL_THICKNESS = 10
    INITIAL_WALL_SHRINK_SPEED = 0.1
    WALL_SHRINK_ACCELERATION = 0.05
    WIN_CONDITION_GEMS = 20
    MAX_STEPS = 1000
    INITIAL_GEMS = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.wall_bounds = [0, 0, 0, 0]
        self.gems = []
        self.particles = []
        self.gems_collected = 0
        self.wall_shrink_speed = 0.0
        self.bounce_vel = 0.0
        self.steps = 0
        self.wall_pulse_phase = 0.0
        self.last_reward = 0.0
        self.game_over = False
        
        # self.reset() is called by the wrapper, no need to call it here.
        # self.validate_implementation() # This is a debug method, not needed in production.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.gems_collected = 0
        self.game_over = False
        
        self.ball_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]
        self.ball_vel = [0, 0]
        
        self.wall_bounds = [
            float(self.WALL_THICKNESS), 
            float(self.SCREEN_WIDTH - self.WALL_THICKNESS),
            float(self.WALL_THICKNESS),
            float(self.SCREEN_HEIGHT - self.WALL_THICKNESS)
        ]
        
        self.wall_shrink_speed = self.INITIAL_WALL_SHRINK_SPEED
        self.bounce_vel = self.INITIAL_BOUNCE_VEL
        
        self.gems = []
        self._spawn_gems(self.INITIAL_GEMS)
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.last_reward = 0.0 # Reset per-step reward
        self.steps += 1

        # --- Action Handling ---
        self._handle_input(action)

        # --- Game Logic Updates ---
        self._update_ball_physics()
        self._update_walls()
        self._update_particles()
        
        # --- Collision & Event Handling ---
        gem_collected_this_step = self._handle_gem_collisions()
        wall_collision = self._handle_wall_collisions()
        
        # --- Reward Calculation ---
        reward = 0.1 # Survival reward
        if gem_collected_this_step:
            reward += 1.0
        
        # --- Termination Check ---
        terminated = False
        truncated = False
        if wall_collision:
            reward = -100.0
            terminated = True
            # sfx: player_death
        elif self.gems_collected >= self.WIN_CONDITION_GEMS:
            reward = 100.0
            terminated = True
            # sfx: win_game
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        self.game_over = terminated or truncated
        self.last_reward = reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        # Horizontal Movement
        if movement == 3: # Left
            self.ball_pos[0] -= self.PLAYER_HORIZONTAL_SPEED
        elif movement == 4: # Right
            self.ball_pos[0] += self.PLAYER_HORIZONTAL_SPEED
            
        # Vertical Thrust
        if movement == 1: # Up
            self.ball_vel[1] -= self.PLAYER_VERTICAL_THRUST
        elif movement == 2: # Down
            self.ball_vel[1] += self.PLAYER_VERTICAL_THRUST

    def _update_ball_physics(self):
        # Apply gravity
        self.ball_vel[1] += self.GRAVITY
        
        # Update position
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]
        
        # Dampen horizontal velocity (friction)
        self.ball_vel[0] *= 0.95

    def _update_walls(self):
        if self.steps > 0 and self.steps % 100 == 0:
            self.wall_shrink_speed += self.WALL_SHRINK_ACCELERATION
            
        self.wall_bounds[0] += self.wall_shrink_speed / self.TARGET_FPS
        self.wall_bounds[1] -= self.wall_shrink_speed / self.TARGET_FPS
        self.wall_bounds[2] += self.wall_shrink_speed / self.TARGET_FPS
        self.wall_bounds[3] -= self.wall_shrink_speed / self.TARGET_FPS

        # Prevent walls from crossing
        self.wall_bounds[0] = min(self.wall_bounds[0], self.SCREEN_WIDTH / 2 - 1)
        self.wall_bounds[1] = max(self.wall_bounds[1], self.SCREEN_WIDTH / 2 + 1)
        self.wall_bounds[2] = min(self.wall_bounds[2], self.SCREEN_HEIGHT / 2 - 1)
        self.wall_bounds[3] = max(self.wall_bounds[3], self.SCREEN_HEIGHT / 2 + 1)
        
    def _handle_gem_collisions(self):
        collected_this_step = False
        for gem_pos in self.gems[:]:
            dist_sq = (self.ball_pos[0] - gem_pos[0])**2 + (self.ball_pos[1] - gem_pos[1])**2
            if dist_sq < (self.BALL_RADIUS + self.GEM_RADIUS)**2:
                self.gems.remove(gem_pos)
                self.gems_collected += 1
                self.bounce_vel -= 2.0 # Increase bounce height
                self._spawn_particles(gem_pos, self.COLOR_GEM, 20)
                self.wall_pulse_phase = 0 # Sync wall pulse
                collected_this_step = True
                # sfx: gem_collect
        
        if not self.gems:
            self._spawn_gems(self.INITIAL_GEMS)
            
        return collected_this_step

    def _handle_wall_collisions(self):
        # Bounce off floor/ceiling
        if self.ball_pos[1] + self.BALL_RADIUS >= self.wall_bounds[3]:
            self.ball_pos[1] = self.wall_bounds[3] - self.BALL_RADIUS
            self.ball_vel[1] = self.bounce_vel
            # sfx: bounce
        if self.ball_pos[1] - self.BALL_RADIUS <= self.wall_bounds[2]:
            self.ball_pos[1] = self.wall_bounds[2] + self.BALL_RADIUS
            self.ball_vel[1] *= -0.5 # Weaker bounce off ceiling
            # sfx: bounce
            
        # Bounce off side walls
        if self.ball_pos[0] + self.BALL_RADIUS >= self.wall_bounds[1]:
            self.ball_pos[0] = self.wall_bounds[1] - self.BALL_RADIUS
            self.ball_vel[0] *= -0.8
            # sfx: bounce
        if self.ball_pos[0] - self.BALL_RADIUS <= self.wall_bounds[0]:
            self.ball_pos[0] = self.wall_bounds[0] + self.BALL_RADIUS
            self.ball_vel[0] *= -0.8
            # sfx: bounce

        # Check for crush death
        if (self.ball_pos[0] - self.BALL_RADIUS < self.wall_bounds[0] or
            self.ball_pos[0] + self.BALL_RADIUS > self.wall_bounds[1] or
            self.ball_pos[1] - self.BALL_RADIUS < self.wall_bounds[2] or
            self.ball_pos[1] + self.BALL_RADIUS > self.wall_bounds[3]):
            return True
        
        return False

    def _spawn_gems(self, count):
        for _ in range(count):
            # Ensure gems spawn within the inner 80% of the wall bounds
            margin_x = (self.wall_bounds[1] - self.wall_bounds[0]) * 0.1
            margin_y = (self.wall_bounds[3] - self.wall_bounds[2]) * 0.1
            
            x = self.np_random.uniform(self.wall_bounds[0] + margin_x, self.wall_bounds[1] - margin_x)
            y = self.np_random.uniform(self.wall_bounds[2] + margin_y, self.wall_bounds[3] - margin_y)
            self.gems.append([x, y])

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95
            p['vel'][1] *= 0.95
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.gems_collected,
            "steps": self.steps,
            "bounce_height": -self.bounce_vel,
        }

    def _render_game(self):
        self._render_walls()
        self._render_gems()
        self._render_particles()
        self._render_ball()

    def _render_walls(self):
        self.wall_pulse_phase += 0.1
        pulse = (math.sin(self.wall_pulse_phase) + 1) / 2 # 0 to 1
        
        r = self.COLOR_WALL_BASE[0]
        g = int(self.COLOR_WALL_BASE[1] * (0.5 + pulse * 0.5))
        b = int(self.COLOR_WALL_BASE[2] * (0.8 + pulse * 0.2))
        pulsing_color = (r, g, b)
        
        bounds = [int(b) for b in self.wall_bounds]
        
        # Draw filled rectangles to create thick walls
        pygame.draw.rect(self.screen, pulsing_color, (0, 0, self.SCREEN_WIDTH, bounds[2])) # Top
        pygame.draw.rect(self.screen, pulsing_color, (0, bounds[3], self.SCREEN_WIDTH, self.SCREEN_HEIGHT - bounds[3])) # Bottom
        pygame.draw.rect(self.screen, pulsing_color, (0, 0, bounds[0], self.SCREEN_HEIGHT)) # Left
        pygame.draw.rect(self.screen, pulsing_color, (bounds[1], 0, self.SCREEN_WIDTH - bounds[1], self.SCREEN_HEIGHT)) # Right

    def _render_gems(self):
        for pos in self.gems:
            self._draw_glow_circle(self.screen, self.COLOR_GEM, (int(pos[0]), int(pos[1])), self.GEM_RADIUS, 150)

    def _render_ball(self):
        pos_int = (int(self.ball_pos[0]), int(self.ball_pos[1]))
        self._draw_glow_circle(self.screen, self.COLOR_BALL, pos_int, self.BALL_RADIUS, 200)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30.0))))
            color = p['color']
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*color, alpha))

    def _render_ui(self):
        gem_text = self.font_main.render(f"GEMS: {self.gems_collected}/{self.WIN_CONDITION_GEMS}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (15, 15))
        
        bounce_text = self.font_main.render(f"BOUNCE: {int(-self.bounce_vel)}", True, self.COLOR_TEXT)
        self.screen.blit(bounce_text, (self.SCREEN_WIDTH - bounce_text.get_width() - 15, 15))

    def _draw_glow_circle(self, surface, color, center, radius, glow_strength):
        for i in range(radius, 0, -1):
            alpha = int(glow_strength * (1 - (i / radius))**2)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(surface, center[0], center[1], i + 3, (*color, alpha // 4))
        
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)

    def close(self):
        pygame.font.quit()
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # It will not run in a headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bouncing Gem Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        
        # Manual control mapping
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Convert observation back to a surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        clock.tick(GameEnv.TARGET_FPS)
        
    env.close()