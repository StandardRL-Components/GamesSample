
# Generated: 2025-08-28T05:25:38.518140
# Source Brief: brief_02623.md
# Brief Index: 2623

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: ↑↓←→ to move. Hold Shift to drift/dodge. Press Space to fire your weapon."
    )

    game_description = (
        "Survive waves of zombies in a top-down arena shooter. Kill zombies to score points and clear waves to win."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 3000  # Approx 100 seconds
    MAX_WAVES = 5

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_WALL = (50, 50, 60)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50)
    COLOR_ZOMBIE = (255, 50, 50)
    COLOR_ZOMBIE_GLOW = (255, 50, 50, 50)
    COLOR_BULLET = (255, 255, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_GREEN = (50, 200, 50)
    COLOR_HEALTH_RED = (200, 50, 50)

    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 3.5
    PLAYER_MAX_HEALTH = 100
    PLAYER_ACCEL = 0.6
    PLAYER_FRICTION = 0.90
    PLAYER_DRIFT_SPEED_MULT = 1.8
    PLAYER_DRIFT_FRICTION = 0.98
    PLAYER_SHOOT_COOLDOWN = 6 # frames

    # Zombie
    ZOMBIE_SIZE = 14
    ZOMBIE_MAX_HEALTH = 20
    ZOMBIE_BASE_SPEED = 0.8
    ZOMBIE_SPEED_WAVE_INC = 0.1
    ZOMBIE_DAMAGE = 15

    # Bullet
    BULLET_SIZE = 3
    BULLET_SPEED = 12.0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_aim_angle = None
        self.shoot_cooldown = None
        self.zombies = None
        self.bullets = None
        self.particles = None
        self.current_wave = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_aim_angle = 0.0
        
        self.shoot_cooldown = 0
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.current_wave = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01 # Small reward for surviving a frame

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self._handle_input(movement, space_held, shift_held)
        
        # Update player
        is_drifting = shift_held and self.player_vel.length() > 0.1
        friction = self.PLAYER_DRIFT_FRICTION if is_drifting else self.PLAYER_FRICTION
        self.player_vel *= friction
        self.player_pos += self.player_vel
        self._keep_player_in_bounds()
        
        # Update bullets and check collisions
        reward += self._update_bullets()
        
        # Update zombies and check collisions
        reward += self._update_zombies()
        
        # Update particles
        self._update_particles()
        
        # Wave progression
        if not self.zombies and not self.game_over:
            reward += 100
            if self.current_wave >= self.MAX_WAVES:
                self.game_won = True
                self.game_over = True
            else:
                self._spawn_wave()
                # sound: wave_cleared.wav
        
        # Check termination conditions
        self.steps += 1
        terminated = False
        if self.player_health <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            self._create_particles(self.player_pos, 50, self.COLOR_PLAYER, 3, 60)
            # sound: player_death.wav
        elif self.game_won:
            reward += 500
            terminated = True
            # sound: game_win.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_aim_angle = math.atan2(move_vec.y, move_vec.x)
            
            speed_mult = self.PLAYER_DRIFT_SPEED_MULT if shift_held else 1.0
            self.player_vel += move_vec * self.PLAYER_ACCEL * speed_mult
            
            if shift_held and self.np_random.random() < 0.3:
                 self._create_particles(self.player_pos, 1, (200,200,255), 2, 10, angle_spread=math.pi, speed_range=(0.5, 1.5))
                 # sound: drift.wav

        # Limit max speed
        max_speed = self.PLAYER_SPEED * (self.PLAYER_DRIFT_SPEED_MULT if shift_held else 1.0)
        if self.player_vel.length() > max_speed:
            self.player_vel.scale_to_length(max_speed)

        if space_held and self.shoot_cooldown == 0:
            self._fire_bullet()
            return -0.2 # Small penalty for firing
        return 0

    def _fire_bullet(self):
        self.shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
        bullet_vel = pygame.Vector2(math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)) * self.BULLET_SPEED
        spawn_pos = self.player_pos + bullet_vel.normalize() * (self.PLAYER_SIZE + 5)
        self.bullets.append({'pos': spawn_pos, 'vel': bullet_vel})
        self._create_particles(spawn_pos, 5, self.COLOR_BULLET, 2, 10, angle_spread=0.5, speed_range=(1,2))
        # sound: shoot.wav

    def _update_bullets(self):
        reward = 0
        for bullet in self.bullets[:]:
            bullet['pos'] += bullet['vel']
            
            # Zombie collision
            hit = False
            for zombie in self.zombies[:]:
                if bullet['pos'].distance_to(zombie['pos']) < self.ZOMBIE_SIZE:
                    zombie['health'] -= 10
                    reward += 1
                    self._create_particles(bullet['pos'], 10, self.COLOR_ZOMBIE, 1.5, 20)
                    # sound: zombie_hit.wav
                    if zombie['health'] <= 0:
                        self.zombies.remove(zombie)
                        self.score += 10
                        reward += 10
                        self._create_particles(zombie['pos'], 30, self.COLOR_ZOMBIE, 2.5, 40)
                        # sound: zombie_death.wav
                    hit = True
                    break
            
            if hit or not (0 < bullet['pos'].x < self.WIDTH and 0 < bullet['pos'].y < self.HEIGHT):
                self.bullets.remove(bullet)
        return reward

    def _update_zombies(self):
        reward = 0
        for zombie in self.zombies:
            direction = (self.player_pos - zombie['pos'])
            if direction.length() > 0:
                direction.normalize_ip()
            zombie['pos'] += direction * zombie['speed']
            
            if zombie['pos'].distance_to(self.player_pos) < (self.PLAYER_SIZE + self.ZOMBIE_SIZE) / 2:
                self.player_health -= self.ZOMBIE_DAMAGE
                reward -= 5
                self.player_vel += (self.player_pos - zombie['pos']).normalize() * 3 # Knockback
                self._create_particles(self.player_pos, 20, self.COLOR_PLAYER, 2, 30)
                # sound: player_hit.wav
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        self.current_wave += 1
        num_zombies = 3 + self.current_wave * 2
        zombie_speed = self.ZOMBIE_BASE_SPEED + (self.current_wave - 1) * self.ZOMBIE_SPEED_WAVE_INC
        
        for _ in range(num_zombies):
            # Spawn on edges
            edge = self.np_random.integers(4)
            if edge == 0: x, y = self.np_random.integers(-20, 0), self.np_random.integers(self.HEIGHT)
            elif edge == 1: x, y = self.np_random.integers(self.WIDTH, self.WIDTH + 20), self.np_random.integers(self.HEIGHT)
            elif edge == 2: x, y = self.np_random.integers(self.WIDTH), self.np_random.integers(-20, 0)
            else: x, y = self.np_random.integers(self.WIDTH), self.np_random.integers(self.HEIGHT, self.HEIGHT + 20)
            
            self.zombies.append({
                'pos': pygame.Vector2(x, y),
                'health': self.ZOMBIE_MAX_HEALTH,
                'speed': zombie_speed * self.np_random.uniform(0.9, 1.1)
            })

    def _keep_player_in_bounds(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw particles first (background layer)
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            p['color'] = p['base_color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Draw zombies
        for z in self.zombies:
            pos_int = (int(z['pos'].x), int(z['pos'].y))
            self._draw_glowing_circle(self.screen, pos_int, self.ZOMBIE_SIZE, self.COLOR_ZOMBIE, self.COLOR_ZOMBIE_GLOW)
            # Health bar
            health_pct = max(0, z['health'] / self.ZOMBIE_MAX_HEALTH)
            bar_w = self.ZOMBIE_SIZE * 2
            bar_h = 4
            bar_x = pos_int[0] - bar_w / 2
            bar_y = pos_int[1] - self.ZOMBIE_SIZE - bar_h - 2
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, bar_w * health_pct, bar_h))

        # Draw bullets
        for b in self.bullets:
            pos_int = (int(b['pos'].x), int(b['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_BULLET, pos_int, self.BULLET_SIZE)
            pygame.draw.circle(self.screen, self.COLOR_BULLET + (100,), pos_int, self.BULLET_SIZE * 2)

        # Draw player if not dead
        if self.player_health > 0:
            pos_int = (int(self.player_pos.x), int(self.player_pos.y))
            self._draw_glowing_circle(self.screen, pos_int, self.PLAYER_SIZE, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
            # Aiming indicator
            end_pos = self.player_pos + pygame.Vector2(math.cos(self.player_aim_angle), math.sin(self.player_aim_angle)) * (self.PLAYER_SIZE + 4)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, pos_int, (int(end_pos.x), int(end_pos.y)), 2)
            
    def _render_ui(self):
        # Score and Wave
        self._draw_text(f"SCORE: {int(self.score)}", (10, 10), self.font_small)
        self._draw_text(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", (10, 35), self.font_small)
        
        # Player Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_w, bar_h = 150, 20
        bar_x, bar_y = self.WIDTH - bar_w - 10, 10
        pygame.draw.rect(self.screen, (50,50,50), (bar_x-2, bar_y-2, bar_w+4, bar_h+4))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_x, bar_y, bar_w * health_pct, bar_h))
        self._draw_text("HEALTH", (bar_x + bar_w/2, bar_y + bar_h/2), self.font_small, center=True)

        # Game Over / Win Message
        if self.game_over:
            message = "YOU WON!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.COLOR_ZOMBIE
            self._draw_text(message, (self.WIDTH/2, self.HEIGHT/2), self.font_large, color=color, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center: text_rect.center = pos
        else: text_rect.topleft = pos
        
        # Draw outline for readability
        outline_surf = font.render(text, True, (20,20,20))
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
             self.screen.blit(outline_surf, (text_rect.x + dx, text_rect.y + dy))
        
        self.screen.blit(text_surf, text_rect)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color):
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius * 1.8), glow_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(radius * 1.5), glow_color)
        pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], radius, color)

    def _create_particles(self, pos, count, color, radius, lifespan, angle_spread=2*math.pi, speed_range=(1,4)):
        for _ in range(count):
            angle = self.np_random.uniform(-angle_spread/2, angle_spread/2) + self.np_random.uniform(0, 2*math.pi)
            speed = self.np_random.uniform(*speed_range)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(radius*0.5, radius),
                'lifespan': self.np_random.integers(lifespan//2, lifespan),
                'max_lifespan': lifespan,
                'base_color': color
            })
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "player_health": self.player_health,
            "zombies_remaining": len(self.zombies),
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()