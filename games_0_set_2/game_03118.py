
# Generated: 2025-08-27T22:25:06.711111
# Source Brief: brief_03118.md
# Brief Index: 3118

        
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
        "Controls: ←→ to move, ↑ to jump. Hold Shift to aim up. Press Space to fire."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "Control a jumping, shooting robot in a side-scrolling arena to defeat 15 enemy robots."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLATFORM_Y = self.HEIGHT - 40
        self.MAX_STEPS = 1500
        self.TOTAL_ENEMIES = 15

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 45)
        self.COLOR_PLATFORM = (80, 80, 100)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GUN = (200, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 200, 50)
        self.COLOR_ENEMY_PROJ = (255, 120, 50)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (50, 255, 50)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        
        # Physics and game constants
        self.GRAVITY = 0.5
        self.PLAYER_SPEED = 5
        self.JUMP_STRENGTH = -10
        self.PLAYER_SHOOT_COOLDOWN = 6 # frames
        self.ENEMY_SHOOT_COOLDOWN_MIN = 60
        self.ENEMY_SHOOT_COOLDOWN_MAX = 120
        self.PROJ_SPEED = 8

        # Initialize state variables
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.screen_shake = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.screen_shake = 0

        self.player = {
            "rect": pygame.Rect(self.WIDTH // 2 - 15, self.PLATFORM_Y - 30, 30, 30),
            "vy": 0,
            "on_ground": True,
            "health": 3,
            "max_health": 3,
            "shoot_cooldown": 0,
            "last_move_dir": 1, # 1 for right, -1 for left
            "hit_timer": 0
        }

        self.enemies = []
        for i in range(self.TOTAL_ENEMIES):
            side = random.choice([-1, 1])
            x = self.WIDTH // 2 + side * random.randint(100, self.WIDTH // 2 - 30)
            self.enemies.append({
                "rect": pygame.Rect(x, self.PLATFORM_Y - 24, 24, 24),
                "shoot_timer": self.np_random.integers(30, self.ENEMY_SHOOT_COOLDOWN_MAX)
            })

        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = -0.01  # Small penalty for each step to encourage efficiency
        terminated = False

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_enemies()
            self._update_projectiles()
            
            collision_rewards = self._handle_collisions()
            reward += collision_rewards
            
        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1

        self.steps += 1

        if self.player["health"] <= 0 and not self.game_over:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 100
            self._spawn_explosion(self.player["rect"].center, 40, self.COLOR_PLAYER)

        if len(self.enemies) == 0 and not self.game_over:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100

        if self.steps >= self.MAX_STEPS and not self.game_over:
            self.game_over = True
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal Movement
        player_vx = 0
        if movement == 3: # Left
            player_vx = -self.PLAYER_SPEED
            self.player["last_move_dir"] = -1
        elif movement == 4: # Right
            player_vx = self.PLAYER_SPEED
            self.player["last_move_dir"] = 1
        self.player["rect"].x += player_vx
        
        # Jumping
        if movement == 1 and self.player["on_ground"]:
            self.player["vy"] = self.JUMP_STRENGTH
            self.player["on_ground"] = False
            # // SFX: Jump/Thruster
            for _ in range(10):
                self.particles.append(self._create_particle(
                    (self.player["rect"].centerx, self.player["rect"].bottom),
                    vel=[(random.random() - 0.5) * 2, random.random() * 2],
                    life=10, radius=random.randint(2,4), color=(200,200,200)
                ))

        # Shooting
        if space_held and self.player["shoot_cooldown"] == 0:
            self.player["shoot_cooldown"] = self.PLAYER_SHOOT_COOLDOWN
            
            if shift_held: # Aim up
                vx = self.PROJ_SPEED * 0.707 * self.player["last_move_dir"]
                vy = -self.PROJ_SPEED * 0.707
            else: # Aim horizontal
                vx = self.PROJ_SPEED * self.player["last_move_dir"]
                vy = 0
            
            proj_rect = pygame.Rect(self.player["rect"].centerx, self.player["rect"].centery - 2, 8, 4)
            self.player_projectiles.append({"rect": proj_rect, "vx": vx, "vy": vy})
            # // SFX: Laser shoot

    def _update_player(self):
        # Apply gravity
        self.player["vy"] += self.GRAVITY
        self.player["rect"].y += int(self.player["vy"])

        # Ground collision
        if self.player["rect"].bottom >= self.PLATFORM_Y:
            self.player["rect"].bottom = self.PLATFORM_Y
            self.player["vy"] = 0
            self.player["on_ground"] = True

        # Screen boundaries
        self.player["rect"].left = max(0, self.player["rect"].left)
        self.player["rect"].right = min(self.WIDTH, self.player["rect"].right)
        
        # Cooldowns
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1
        if self.player["hit_timer"] > 0:
            self.player["hit_timer"] -= 1

    def _update_enemies(self):
        for enemy in self.enemies:
            # Simple AI: move towards player
            if enemy["rect"].centerx < self.player["rect"].centerx:
                enemy["rect"].x += 1
            elif enemy["rect"].centerx > self.player["rect"].centerx:
                enemy["rect"].x -= 1
            
            # Shooting logic
            enemy["shoot_timer"] -= 1
            if enemy["shoot_timer"] <= 0:
                enemy["shoot_timer"] = self.np_random.integers(self.ENEMY_SHOOT_COOLDOWN_MIN, self.ENEMY_SHOOT_COOLDOWN_MAX)
                dx = self.player["rect"].centerx - enemy["rect"].centerx
                dy = self.player["rect"].centery - enemy["rect"].centery
                dist = math.hypot(dx, dy)
                if dist > 0:
                    vx = (dx / dist) * self.PROJ_SPEED
                    vy = (dy / dist) * self.PROJ_SPEED
                    proj_rect = pygame.Rect(enemy["rect"].centerx, enemy["rect"].centery, 8, 8)
                    self.enemy_projectiles.append({"rect": proj_rect, "vx": vx, "vy": vy})
                    # // SFX: Enemy shoot

    def _update_projectiles(self):
        for proj_list in [self.player_projectiles, self.enemy_projectiles]:
            for proj in proj_list[:]:
                proj["rect"].x += int(proj["vx"])
                proj["rect"].y += int(proj["vy"])
                if not self.screen.get_rect().colliderect(proj["rect"]):
                    proj_list.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] -= 0.1
            if p['life'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj["rect"].colliderect(enemy["rect"]):
                    self.enemies.remove(enemy)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    reward += 10
                    self.score += 1
                    self._spawn_explosion(enemy["rect"].center, 20, self.COLOR_ENEMY)
                    # // SFX: Explosion
                    break
        
        # Enemy projectiles vs player
        if self.player["hit_timer"] == 0:
            for proj in self.enemy_projectiles[:]:
                if self.player["rect"].colliderect(proj["rect"]):
                    self.enemy_projectiles.remove(proj)
                    self.player["health"] -= 1
                    self.player["hit_timer"] = 30 # 1 second invulnerability
                    reward -= 1
                    self.screen_shake = 10
                    # // SFX: Player hit
                    break
        return reward

    def _spawn_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = random.random() * 2 * math.pi
            speed = random.random() * 4 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append(self._create_particle(
                pos, vel, life=random.randint(15, 30),
                radius=random.randint(3, 7), color=color
            ))

    def _create_particle(self, pos, vel, life, radius, color):
        return {
            'pos': list(pos), 'vel': vel, 'life': life,
            'radius': radius, 'color': color
        }
    
    def _get_observation(self):
        offset_x, offset_y = 0, 0
        if self.screen_shake > 0:
            offset_x = random.randint(-4, 4)
            offset_y = random.randint(-4, 4)

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._render_background(offset_x, offset_y)
        
        # Render all game elements
        self._render_game(offset_x, offset_y)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, ox, oy):
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i + ox, 0), (i + ox, self.HEIGHT))
        for i in range(0, self.HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i + oy), (self.WIDTH, i + oy))
        
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (0 + ox, self.PLATFORM_Y + oy, self.WIDTH, self.HEIGHT - self.PLATFORM_Y))

    def _render_game(self, ox, oy):
        # Render particles
        for p in self.particles:
            pos = (int(p['pos'][0] + ox), int(p['pos'][1] + oy))
            r = int(p['radius'])
            if r > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], r, p['color'])
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], r, p['color'])

        # Render projectiles
        for proj in self.player_projectiles:
            rect = proj["rect"].move(ox, oy)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, rect, border_radius=2)
        for proj in self.enemy_projectiles:
            rect = proj["rect"].move(ox, oy)
            pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, 4, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, 4, self.COLOR_ENEMY_PROJ)

        # Render enemies
        for enemy in self.enemies:
            rect = enemy["rect"].move(ox, oy)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=4)
        
        # Render player
        if self.player["health"] > 0:
            player_color = self.COLOR_PLAYER
            if self.player["hit_timer"] > 0 and (self.player["hit_timer"] // 3) % 2 == 0:
                player_color = (255, 255, 255) # Flash white when hit
            
            rect = self.player["rect"].move(ox, oy)
            pygame.draw.rect(self.screen, player_color, rect, border_radius=4)
            
            # Draw gun
            gun_y = rect.centery
            gun_w = 10
            gun_h = 4
            if self.player["last_move_dir"] > 0:
                gun_x = rect.right
            else:
                gun_x = rect.left - gun_w
            
            gun_rect = pygame.Rect(gun_x, gun_y - gun_h//2, gun_w, gun_h)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_GUN, gun_rect, border_radius=2)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player["health"] / self.player["max_health"]
        bar_width = 150
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))

        # Enemies Remaining
        enemies_text = self.font_small.render(f"ENEMIES: {len(self.enemies)}/{self.TOTAL_ENEMIES}", True, self.COLOR_TEXT)
        self.screen.blit(enemies_text, (self.WIDTH - enemies_text.get_width() - 10, 10))

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, self.HEIGHT - score_text.get_height() - 10))

        # Game Over / Win Text
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player["health"],
            "enemies_left": len(self.enemies),
        }

    def close(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # To display the game, we need to create a Pygame window
    pygame.display.set_caption("Robot Arena")
    real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    # Game loop
    while not done:
        # Simple human controls for testing
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        # No down action for human player
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    env.close()