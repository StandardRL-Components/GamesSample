
# Generated: 2025-08-27T20:35:39.371170
# Source Brief: brief_02512.md
# Brief Index: 2512

        
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
        "Controls: ←→ to rotate. Hold space to fire your weapon."
    )

    game_description = (
        "Rotate your ship and blast asteroids to survive the onslaught in this minimalist arcade shooter. Destroy all 20 asteroids to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_ASTEROID = (150, 150, 150)
        self.COLOR_ASTEROID_OUTLINE = (120, 120, 120)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_PARTICLE = (255, 220, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (50, 255, 50)

        # Game parameters
        self.MAX_STEPS = 1500  # Approx 50 seconds at 30fps
        self.INITIAL_LIVES = 3
        self.INITIAL_ASTEROIDS = 20
        self.PLAYER_ROTATION_SPEED = 6
        self.PLAYER_SIZE = 12
        self.PLAYER_INVINCIBILITY_STEPS = 90  # 3 seconds
        self.PROJECTILE_SPEED = 10
        self.PROJECTILE_LIFESPAN = 40  # steps
        self.FIRE_COOLDOWN_STEPS = 6  # steps
        self.ASTEROID_MIN_RADIUS = 10
        self.ASTEROID_MAX_RADIUS = 25
        self.ASTEROID_MIN_SPEED = 0.5
        self.ASTEROID_MAX_SPEED = 2.0
        
        # Rewards
        self.REWARD_STEP_PENALTY = -0.01
        self.REWARD_ASTEROID_DESTROYED = 1.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSS = -100.0

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.game_font = pygame.font.Font(None, 36)
        self.end_font = pygame.font.Font(None, 72)

        # --- State Variables ---
        self.player = {}
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.win = False
        self.fire_cooldown = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        self.win = False

        self.projectiles.clear()
        self.particles.clear()
        
        self._spawn_player()
        self._spawn_asteroids()
        self._spawn_stars()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        reward = self.REWARD_STEP_PENALTY
        
        if not self.game_over:
            self._handle_input(movement, space_held)
            self._update_game_state()
            reward += self._check_collisions()

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        
        if terminated:
            self.game_over = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Rotate player
        if movement == 3:  # Left
            self.player['angle'] -= self.PLAYER_ROTATION_SPEED
        elif movement == 4:  # Right
            self.player['angle'] += self.PLAYER_ROTATION_SPEED
        self.player['angle'] %= 360

        # Fire projectile
        if space_held and self.fire_cooldown <= 0:
            # // sfx: laser fire
            angle_rad = math.radians(self.player['angle'] - 90)
            nose_offset = pygame.Vector2(
                math.cos(angle_rad) * self.PLAYER_SIZE,
                math.sin(angle_rad) * self.PLAYER_SIZE
            )
            start_pos = self.player['pos'] + nose_offset
            velocity = pygame.Vector2(
                math.cos(angle_rad) * self.PROJECTILE_SPEED,
                math.sin(angle_rad) * self.PROJECTILE_SPEED
            )
            self.projectiles.append({
                'pos': start_pos,
                'vel': velocity,
                'lifespan': self.PROJECTILE_LIFESPAN
            })
            self.fire_cooldown = self.FIRE_COOLDOWN_STEPS

    def _update_game_state(self):
        # Update timers
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
        if self.player['invincible_timer'] > 0:
            self.player['invincible_timer'] -= 1

        # Move projectiles
        for p in self.projectiles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
        self.projectiles = [p for p in self.projectiles if p['lifespan'] > 0]

        # Move asteroids
        for a in self.asteroids:
            a['pos'] += a['vel']
            a['pos'].x %= self.SCREEN_WIDTH
            a['pos'].y %= self.SCREEN_HEIGHT
        
        # Update particles
        for p in self.particles:
            p['lifespan'] -= 1
            p['radius'] -= 0.5
        self.particles = [p for p in self.particles if p['lifespan'] > 0 and p['radius'] > 0]

    def _check_collisions(self):
        reward = 0
        
        # Projectiles vs Asteroids
        projectiles_to_remove = []
        asteroids_to_remove = []
        for i, p in enumerate(self.projectiles):
            for j, a in enumerate(self.asteroids):
                if p['pos'].distance_to(a['pos']) < a['radius']:
                    if i not in projectiles_to_remove:
                        projectiles_to_remove.append(i)
                    if j not in asteroids_to_remove:
                        asteroids_to_remove.append(j)
                        reward += self.REWARD_ASTEROID_DESTROYED
                        self.score += int(self.ASTEROID_MAX_RADIUS - a['radius'] + 10)
                        self._create_explosion(a['pos'], a['radius'])
                        # // sfx: asteroid explosion
        
        # Remove collided entities
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
        self.asteroids = [a for j, a in enumerate(self.asteroids) if j not in asteroids_to_remove]

        # Asteroids vs Player
        if self.player['invincible_timer'] <= 0:
            for a in self.asteroids:
                if self.player['pos'].distance_to(a['pos']) < a['radius'] + self.PLAYER_SIZE * 0.5:
                    self.lives -= 1
                    self._create_explosion(self.player['pos'], self.PLAYER_SIZE * 2)
                    # // sfx: player explosion
                    if self.lives > 0:
                        self._spawn_player()
                    else:
                        self.game_over = True
                    break
        return reward

    def _check_termination(self):
        if not self.asteroids and not self.win:
            self.win = True
            return True, self.REWARD_WIN
        if self.lives <= 0:
            return True, self.REWARD_LOSS
        if self.steps >= self.MAX_STEPS:
            return True, 0.0
        return False, 0.0

    def _spawn_player(self):
        self.player = {
            'pos': pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2),
            'angle': 0,
            'invincible_timer': self.PLAYER_INVINCIBILITY_STEPS
        }

    def _spawn_asteroids(self):
        self.asteroids.clear()
        safe_zone_radius = self.ASTEROID_MAX_RADIUS * 3
        for _ in range(self.INITIAL_ASTEROIDS):
            while True:
                pos = pygame.Vector2(
                    self.np_random.uniform(0, self.SCREEN_WIDTH),
                    self.np_random.uniform(0, self.SCREEN_HEIGHT)
                )
                if pos.distance_to(self.player['pos']) > safe_zone_radius:
                    break
            
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(self.ASTEROID_MIN_SPEED, self.ASTEROID_MAX_SPEED)
            vel = pygame.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
            radius = self.np_random.uniform(self.ASTEROID_MIN_RADIUS, self.ASTEROID_MAX_RADIUS)
            
            num_points = self.np_random.integers(7, 12)
            points = []
            for i in range(num_points):
                angle_offset = (360 / num_points) * i
                dist_offset = self.np_random.uniform(0.7, 1.1)
                point_radius = radius * dist_offset
                p_angle = math.radians(angle_offset)
                points.append(
                    (math.cos(p_angle) * point_radius, math.sin(p_angle) * point_radius)
                )

            self.asteroids.append({'pos': pos, 'vel': vel, 'radius': radius, 'points': points})

    def _spawn_stars(self):
        self.stars.clear()
        for _ in range(100):
            self.stars.append((
                self.np_random.uniform(0, self.SCREEN_WIDTH),
                self.np_random.uniform(0, self.SCREEN_HEIGHT)
            ))

    def _create_explosion(self, pos, size):
        num_particles = int(size)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle))) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifespan': self.np_random.integers(15, 30)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render stars
        for x, y in self.stars:
            pygame.gfxdraw.pixel(self.screen, int(x), int(y), self.COLOR_STAR)

        # Render asteroids
        for a in self.asteroids:
            world_points = [(p[0] + a['pos'].x, p[1] + a['pos'].y) for p in a['points']]
            pygame.gfxdraw.aapolygon(self.screen, world_points, self.COLOR_ASTEROID_OUTLINE)
            pygame.gfxdraw.filled_polygon(self.screen, world_points, self.COLOR_ASTEROID)

        # Render projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(p['pos'].x), int(p['pos'].y)), 2)
        
        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 30.0))))
            color = (self.COLOR_PARTICLE[0], self.COLOR_PARTICLE[1], self.COLOR_PARTICLE[2], alpha)
            if p['radius'] > 0:
                 pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

        # Render player
        if self.lives > 0:
            is_invincible_blink = self.player['invincible_timer'] > 0 and (self.steps // 4) % 2 == 0
            if not is_invincible_blink:
                angle_rad = math.radians(self.player['angle'] - 90)
                p1 = self.player['pos'] + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * self.PLAYER_SIZE
                p2 = self.player['pos'] + pygame.Vector2(math.cos(angle_rad + 2.5), math.sin(angle_rad + 2.5)) * self.PLAYER_SIZE * 0.8
                p3 = self.player['pos'] + pygame.Vector2(math.cos(angle_rad - 2.5), math.sin(angle_rad - 2.5)) * self.PLAYER_SIZE * 0.8
                
                points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
                
                # Glow effect
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.game_font.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.lives):
            p1 = (self.SCREEN_WIDTH - 30 - i * 25, 20)
            p2 = (self.SCREEN_WIDTH - 40 - i * 25, 35)
            p3 = (self.SCREEN_WIDTH - 20 - i * 25, 35)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1, p2, p3], 1)

        # Game Over / Win message
        if self.game_over:
            if self.win:
                end_text = self.end_font.render("YOU WIN!", True, self.COLOR_WIN)
            else:
                end_text = self.end_font.render("GAME OVER", True, self.COLOR_GAMEOVER)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "asteroids_remaining": len(self.asteroids),
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Human Play Loop ---
    # This demonstrates how to map keyboard inputs to the MultiDiscrete action space.
    
    # Default action: do nothing
    action = np.array([0, 0, 0]) 
    
    # Pygame window for human play
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Asteroid Blaster")
    clock = pygame.time.Clock()

    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Get key presses
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Map keys to MultiDiscrete action space
        # actions[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        # actions[1]: Space button (0=released, 1=held)
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # actions[2]: Shift button (0=released, 1=held) - Unused in this game
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Maintain a consistent frame rate for human play
        clock.tick(30)

        if terminated:
            print(f"Game Over! Final Info: {info}")
            pygame.time.wait(3000) # Pause for 3 seconds before closing

    env.close()