
# Generated: 2025-08-27T14:51:20.830314
# Source Brief: brief_00810.md
# Brief Index: 810

        
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
        "Controls: Arrow keys to move. Space to shoot in your last moved direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Hunt down swarming bugs in a top-down arcade environment before time runs out. Clear all 25 bugs to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.NUM_BUGS = 25
        
        # Player constants
        self.PLAYER_SIZE = 20
        self.PLAYER_ACCEL = 1.2
        self.PLAYER_FRICTION = 0.90
        
        # Projectile constants
        self.PROJECTILE_SIZE = 6
        self.PROJECTILE_SPEED = 10
        self.FIRE_COOLDOWN = 5  # frames

        # Colors (Bright/High-Contrast for interactive, Muted for BG)
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_PLAYER_GLOW = (255, 120, 120)
        self.COLOR_BUG = (80, 180, 255)
        self.COLOR_BUG_GLOW = (120, 200, 255)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        
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
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.aim_direction = None
        self.bugs = []
        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_shot_step = 0
        
        self.reset()
        
        # self.validate_implementation() # For development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.aim_direction = pygame.Vector2(0, -1) # Default aim up
        
        self.bugs = self._create_bugs()
        self.projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_shot_step = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        reward = -0.01 # Small penalty per step to encourage speed

        self._handle_input(movement, space_held)
        self._update_game_state()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if len(self.bugs) == 0:
                reward += 50 # Win bonus
            else:
                reward -= 50 # Lose penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: # Up
            self.player_vel.y -= self.PLAYER_ACCEL
            self.aim_direction = pygame.Vector2(0, -1)
        elif movement == 2: # Down
            self.player_vel.y += self.PLAYER_ACCEL
            self.aim_direction = pygame.Vector2(0, 1)
        elif movement == 3: # Left
            self.player_vel.x -= self.PLAYER_ACCEL
            self.aim_direction = pygame.Vector2(-1, 0)
        elif movement == 4: # Right
            self.player_vel.x += self.PLAYER_ACCEL
            self.aim_direction = pygame.Vector2(1, 0)

        # Shooting
        if space_held and (self.steps - self.last_shot_step) >= self.FIRE_COOLDOWN:
            # sfx: player_shoot.wav
            proj_pos = self.player_pos.copy()
            proj_vel = self.aim_direction.normalize() * self.PROJECTILE_SPEED
            self.projectiles.append({'pos': proj_pos, 'vel': proj_vel, 'size': self.PROJECTILE_SIZE})
            self.last_shot_step = self.steps

    def _update_game_state(self):
        # Update player
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

        # Update projectiles
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            if not self.screen.get_rect().collidepoint(p['pos']):
                self.projectiles.remove(p)

        # Update bugs
        for bug in self.bugs:
            bug['update_func'](bug)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        step_reward = 0
        for proj in self.projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'].x - proj['size']/2, proj['pos'].y - proj['size']/2, proj['size'], proj['size'])
            for bug in self.bugs[:]:
                if proj_rect.colliderect(bug['rect']):
                    # sfx: bug_squish.wav
                    self._create_particles(bug['pos'], self.COLOR_BUG, 15)
                    self.bugs.remove(bug)
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    self.score += 1
                    step_reward += 1
                    break # Projectile can only hit one bug
        return step_reward

    def _check_termination(self):
        if len(self.bugs) == 0:
            return True # Victory
        if self.steps >= self.MAX_STEPS:
            return True # Time out
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw bugs
        for bug in self.bugs:
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, int(bug['pos'].x), int(bug['pos'].y), int(bug['radius'] + 3), (*self.COLOR_BUG_GLOW, 50))
            pygame.gfxdraw.filled_circle(self.screen, int(bug['pos'].x), int(bug['pos'].y), int(bug['radius']), self.COLOR_BUG)
            pygame.gfxdraw.aacircle(self.screen, int(bug['pos'].x), int(bug['pos'].y), int(bug['radius']), self.COLOR_BUG)

        # Draw player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_GLOW, 80), glow_surf.get_rect(), border_radius=5)
        self.screen.blit(glow_surf, glow_rect.topleft)
        # Player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)

        # Draw projectiles
        for p in self.projectiles:
            rect = pygame.Rect(0, 0, p['size'], p['size'])
            rect.center = (int(p['pos'].x), int(p['pos'].y))
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, rect, border_radius=2)
            
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['lifespan'] / p['max_lifespan']))
            if size > 0:
                surf = pygame.Surface((size, size), pygame.SRCALPHA)
                surf.fill(color)
                self.screen.blit(surf, p['pos'] - pygame.Vector2(size/2, size/2))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = self.font_large.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 10))
        
        # Game Over Message
        if self._check_termination():
            msg = "LEVEL CLEAR!" if len(self.bugs) == 0 else "TIME UP!"
            color = (100, 255, 100) if len(self.bugs) == 0 else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_left": len(self.bugs)
        }

    def _create_bugs(self):
        bugs = []
        for _ in range(self.NUM_BUGS):
            pattern = self.np_random.choice(['linear', 'circular'])
            radius = self.np_random.integers(8, 15)
            pos = pygame.Vector2(
                self.np_random.uniform(radius, self.WIDTH - radius),
                self.np_random.uniform(radius, self.HEIGHT - radius)
            )
            
            if pattern == 'linear':
                p1 = pos
                p2 = pygame.Vector2(
                    self.np_random.uniform(radius, self.WIDTH - radius),
                    self.np_random.uniform(radius, self.HEIGHT - radius)
                )
                speed = self.np_random.uniform(1.0, 2.5)
                bug = {
                    'pos': p1, 'radius': radius, 'speed': speed, 'p1': p1, 'p2': p2,
                    'target': 'p2', 'update_func': self._update_bug_linear
                }
            else: # circular
                center = pos
                orbit_radius = self.np_random.uniform(20, 80)
                angle = self.np_random.uniform(0, 2 * math.pi)
                angular_speed = self.np_random.uniform(0.02, 0.05) * self.np_random.choice([-1, 1])
                bug = {
                    'pos': center, 'radius': radius, 'center': center, 'orbit_radius': orbit_radius,
                    'angle': angle, 'angular_speed': angular_speed, 'update_func': self._update_bug_circular
                }
            bug['rect'] = pygame.Rect(bug['pos'].x - radius, bug['pos'].y - radius, radius*2, radius*2)
            bugs.append(bug)
        return bugs

    def _update_bug_linear(self, bug):
        target_pos = bug[bug['target']]
        direction = (target_pos - bug['pos'])
        if direction.length() < bug['speed']:
            bug['pos'] = target_pos
            bug['target'] = 'p1' if bug['target'] == 'p2' else 'p2'
        else:
            bug['pos'] += direction.normalize() * bug['speed']
        bug['rect'].center = bug['pos']

    def _update_bug_circular(self, bug):
        bug['angle'] += bug['angular_speed']
        bug['pos'].x = bug['center'].x + math.cos(bug['angle']) * bug['orbit_radius']
        bug['pos'].y = bug['center'].y + math.sin(bug['angle']) * bug['orbit_radius']
        bug['rect'].center = bug['pos']

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'color': color, 
                'lifespan': lifespan, 'max_lifespan': lifespan,
                'size': self.np_random.integers(2, 6)
            })

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Bug Hunter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    running = True
    while running:
        # Human input mapping
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation
        # Pygame uses (width, height), numpy uses (height, width)
        # The env returns (height, width, 3), so we need to transpose for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()