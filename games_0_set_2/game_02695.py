
# Generated: 2025-08-28T05:39:07.523829
# Source Brief: brief_02695.md
# Brief Index: 2695

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a zombie horde for 60 seconds by running, jumping, and shooting in a side-scrolling horror environment."
    )

    # Frames auto-advance for this real-time game.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GROUND_Y = self.HEIGHT - 50
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_gameover = pygame.font.SysFont("monospace", 50, bold=True)

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_SKY = (25, 25, 40)
        self.COLOR_MOON = (200, 200, 180)
        self.COLOR_CITY = [(30, 30, 50), (40, 40, 60), (50, 50, 70)]
        self.COLOR_GROUND = (20, 20, 20)
        self.COLOR_PLAYER = (220, 220, 220)
        self.COLOR_ZOMBIE = (40, 90, 40)
        self.COLOR_ZOMBIE_EYE = (255, 50, 50)
        self.COLOR_BULLET = (255, 230, 100)
        self.COLOR_MUZZLE_FLASH = (255, 255, 150)
        self.COLOR_BLOOD = (180, 0, 0)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HIT_FLASH = (150, 0, 0, 100)

        # Game constants
        self.MAX_STEPS = 600 # 60 seconds at 10 steps/sec
        self.PLAYER_SPEED = 5
        self.PLAYER_JUMP_POWER = -12
        self.GRAVITY = 0.6
        self.BULLET_SPEED = 15
        self.INITIAL_ZOMBIE_SPEED = 1.0
        self.MAX_ZOMBIES = 10
        self.ZOMBIE_SPAWN_INTERVAL = 25
        self.INVULNERABILITY_DURATION = 30 # steps
        
        # Initialize state variables
        self.background_buildings = []
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_ammo = None
        self.player_direction = None
        self.on_ground = None
        self.zombies = None
        self.bullets = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.zombie_spawn_timer = None
        self.zombie_speed = None
        self.muzzle_flash_timer = None
        self.last_space_held = None
        self.invulnerability_timer = None
        self.screen_flash_timer = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH // 2, self.GROUND_Y)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 3
        self.player_ammo = 30
        self.player_direction = 1
        self.on_ground = True
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.muzzle_flash_timer = 0
        self.last_space_held = False
        self.invulnerability_timer = 0
        self.screen_flash_timer = 0
        
        if not self.background_buildings:
            for i in range(3): # 3 layers of buildings
                for _ in range(20):
                    w = self.np_random.integers(30, 80)
                    h = self.np_random.integers(50, self.GROUND_Y - 50 - i * 40)
                    x = self.np_random.integers(-self.WIDTH, self.WIDTH * 2)
                    self.background_buildings.append(pygame.Rect(x, self.GROUND_Y - h, w, h))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.1 # Survival reward
        
        self._handle_input(action)
        self._update_player()
        self._update_zombies()
        self._update_bullets()
        self._update_particles()
        
        reward += self._handle_collisions()
        
        self._spawn_zombies()
        
        # Update timers and game state
        self.steps += 1
        self.muzzle_flash_timer = max(0, self.muzzle_flash_timer - 1)
        self.invulnerability_timer = max(0, self.invulnerability_timer - 1)
        self.screen_flash_timer = max(0, self.screen_flash_timer - 1)
        
        if self.steps % 100 == 0:
            self.zombie_speed += 0.2

        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win:
                reward += 50
            else:
                reward -= 10 # Penalty for dying
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
            self.player_direction = -1
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
            self.player_direction = 1
        
        # Keep player on screen
        self.player_pos.x = np.clip(self.player_pos.x, 10, self.WIDTH - 10)

        # --- Jumping ---
        if movement == 1 and self.on_ground: # Up
            self.player_vel.y = self.PLAYER_JUMP_POWER
            self.on_ground = False
            # sfx: jump

        # --- Shooting ---
        if space_pressed and not self.last_space_held and self.player_ammo > 0:
            bullet_pos = self.player_pos + pygame.Vector2(self.player_direction * 20, -20)
            self.bullets.append({'pos': bullet_pos, 'dir': self.player_direction})
            self.player_ammo -= 1
            self.muzzle_flash_timer = 3
            # sfx: shoot
        self.last_space_held = space_pressed
        
    def _update_player(self):
        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel
        
        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            self.on_ground = True

    def _update_zombies(self):
        for z in self.zombies:
            direction = 1 if self.player_pos.x > z['pos'].x else -1
            z['pos'].x += direction * self.zombie_speed

    def _update_bullets(self):
        self.bullets = [b for b in self.bullets if 0 < b['pos'].x < self.WIDTH]
        for b in self.bullets:
            b['pos'].x += b['dir'] * self.BULLET_SPEED

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.2 # Particle gravity
            p['life'] -= 1

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0 and len(self.zombies) < self.MAX_ZOMBIES:
            spawn_x = -20 if self.np_random.random() < 0.5 else self.WIDTH + 20
            self.zombies.append({'pos': pygame.Vector2(spawn_x, self.GROUND_Y)})
            self.zombie_spawn_timer = self.ZOMBIE_SPAWN_INTERVAL
        
        # Despawn old zombies if over limit
        while len(self.zombies) > self.MAX_ZOMBIES:
            self.zombies.pop(0)

    def _handle_collisions(self):
        reward = 0
        
        # Bullets and Zombies
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 40, 20, 40)
        
        zombies_to_remove = []
        bullets_to_remove = []
        
        for i, z in enumerate(self.zombies):
            zombie_rect = pygame.Rect(z['pos'].x - 10, z['pos'].y - 40, 20, 40)
            
            # Player-Zombie collision
            if player_rect.colliderect(zombie_rect) and self.invulnerability_timer == 0:
                self.player_health -= 1
                self.invulnerability_timer = self.INVULNERABILITY_DURATION
                self.screen_flash_timer = 5
                reward -= 5
                # sfx: player_hit
                if i not in zombies_to_remove:
                    zombies_to_remove.append(i) # Zombie is removed on contact

            # Bullet-Zombie collision
            for j, b in enumerate(self.bullets):
                bullet_rect = pygame.Rect(b['pos'].x - 2, b['pos'].y - 2, 4, 4)
                if zombie_rect.colliderect(bullet_rect):
                    if i not in zombies_to_remove:
                        zombies_to_remove.append(i)
                    if j not in bullets_to_remove:
                        bullets_to_remove.append(j)
                    reward += 1
                    # sfx: zombie_die
                    
                    # Create blood particles
                    for _ in range(15):
                        self.particles.append({
                            'pos': z['pos'] + pygame.Vector2(0, -20),
                            'vel': pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-4, 1)),
                            'life': self.np_random.integers(15, 30),
                            'size': self.np_random.integers(2, 5)
                        })

        self.zombies = [z for i, z in enumerate(self.zombies) if i not in sorted(zombies_to_remove, reverse=True)]
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in sorted(bullets_to_remove, reverse=True)]
        
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            self.win = False
            return True
        if self.steps >= self.MAX_STEPS:
            self.win = True
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_SKY)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        # Moon
        pygame.gfxdraw.aacircle(self.screen, 100, 80, 40, self.COLOR_MOON)
        pygame.gfxdraw.filled_circle(self.screen, 100, 80, 40, self.COLOR_MOON)
        
        # Cityscape
        for i, color in enumerate(self.COLOR_CITY):
            for building in self.background_buildings:
                if i == 0 and building.height < self.GROUND_Y - 150:
                    pygame.draw.rect(self.screen, color, building)
                elif i == 1 and self.GROUND_Y - 150 <= building.height < self.GROUND_Y - 90:
                    pygame.draw.rect(self.screen, color, building)
                elif i == 2 and building.height >= self.GROUND_Y - 90:
                    pygame.draw.rect(self.screen, color, building)
        
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_game(self):
        # Player
        player_rect = pygame.Rect(int(self.player_pos.x - 10), int(self.player_pos.y - 40), 20, 40)
        
        # Bobbing animation
        if self.on_ground and (self.player_vel.x != 0 or self.steps % 10 < 5):
            player_rect.y -= abs(math.sin(self.steps * 0.5)) * 2
        
        # Invulnerability flash
        if self.invulnerability_timer > 0 and self.steps % 4 < 2:
            pass # Don't draw player
        else:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            # Gun
            gun_x = player_rect.centerx + self.player_direction * 10
            gun_y = player_rect.centery - 5
            pygame.draw.line(self.screen, self.COLOR_PLAYER, (player_rect.centerx, gun_y), (gun_x, gun_y), 4)

        # Muzzle Flash
        if self.muzzle_flash_timer > 0:
            flash_pos = (int(player_rect.centerx + self.player_direction * 15), int(player_rect.centery - 5))
            size = 10
            points = [
                (flash_pos[0] + size, flash_pos[1]), (flash_pos[0], flash_pos[1] + size),
                (flash_pos[0] - size, flash_pos[1]), (flash_pos[0], flash_pos[1] - size)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_MUZZLE_FLASH, points)

        # Zombies
        for z in self.zombies:
            z_rect = pygame.Rect(int(z['pos'].x - 10), int(z['pos'].y - 40), 20, 40)
            z_rect.y -= abs(math.sin(self.steps * 0.2 + z['pos'].x * 0.1)) * 3 # Shambling
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z_rect)
            # Eyes
            eye_dir = 1 if self.player_pos.x > z['pos'].x else -1
            eye_x = int(z_rect.centerx + eye_dir * 4)
            eye_y = int(z_rect.y + 10)
            pygame.draw.circle(self.screen, self.COLOR_ZOMBIE_EYE, (eye_x, eye_y), 2)
            
        # Bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, (int(b['pos'].x), int(b['pos'].y)), 3)
            
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 30.0))))
            color = (*self.COLOR_BLOOD, alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, p['size'], p['size']))
            self.screen.blit(temp_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])))

    def _render_ui(self):
        # Screen flash on hit
        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.COLOR_HIT_FLASH)
            self.screen.blit(flash_surface, (0, 0))
            
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 10.0 # Assuming 10 steps/sec logic
        time_text = self.font_ui.render(f"TIME: {max(0, time_left):.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Health
        health_text = self.font_ui.render("HP:", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (self.WIDTH - 150, 10))
        for i in range(self.player_health):
            heart_surf = self.font_ui.render("♥", True, self.COLOR_ZOMBIE_EYE)
            self.screen.blit(heart_surf, (self.WIDTH - 100 + i * 25, 8))

        # Ammo
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (self.WIDTH - 150, self.HEIGHT - 30))
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU SURVIVED" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            text_surf = self.font_gameover.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies": len(self.zombies),
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed with display drivers
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'dummy', etc. as appropriate

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action mapping from keyboard ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    pygame.quit()