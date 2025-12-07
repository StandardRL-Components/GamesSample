import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓ to move, ←→ to aim. Press space to shoot and hold shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in a dark, side-scrolling shooter. Manage your ammo and stay alive!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5400 # 3 minutes at 30fps

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (40, 35, 50)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_GUN = (150, 150, 150)
        self.COLOR_ZOMBIE = (50, 150, 50)
        self.COLOR_ZOMBIE_HIT = (255, 255, 255)
        self.COLOR_BULLET = (255, 255, 0)
        self.COLOR_BLOOD = (180, 0, 0)
        self.COLOR_MUZZLE_FLASH = (255, 200, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH = (220, 50, 50)
        self.COLOR_AMMO = (200, 200, 200)
        self.COLOR_BAR_BG = (60, 60, 80)
        
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
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.wave_number = 1
        self.wave_transition_timer = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_health = 0
        self.player_ammo = 0
        self.player_aim_direction = 1
        self.is_reloading = False
        self.reload_timer = 0
        self.shot_cooldown = 0
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.bob_offset = 0
        self.background_buildings = []

        # self.reset() is called by the wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player constants
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 30
        self.PLAYER_SPEED = 5
        self.PLAYER_RELOAD_TIME = 60 # 2 seconds at 30fps
        self.PLAYER_SHOT_COOLDOWN = 6 # 5 shots per second
        self.PLAYER_WIDTH, self.PLAYER_HEIGHT = 20, 40
        self.PLAYER_X_POS = self.WIDTH // 4

        # Zombie constants
        self.ZOMBIE_MAX_HEALTH = 30
        self.ZOMBIE_DAMAGE = 10
        self.ZOMBIE_HIT_COOLDOWN = 15 # Can hit player every 0.5s
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.wave_number = 1
        self.wave_transition_timer = 0
        
        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.HEIGHT - 80)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_aim_direction = 1
        
        self.is_reloading = False
        self.reload_timer = 0
        self.shot_cooldown = 0
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self._spawn_wave()
        self._generate_background()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over and not self.game_won:
            self.steps += 1
            self.bob_offset = math.sin(self.steps * 0.2) * 2

            reward += self._handle_actions(action)
            reward += self._update_game_state()

            if self.player_health <= 0:
                self.game_over = True
                reward -= 100
                # Sound effect placeholder: # game over
            
            if self.wave_number > 5:
                self.game_won = True
                reward += 100 # Final win bonus, separate from wave clear bonus
                # Sound effect placeholder: # victory fanfare

            if self.steps >= self.MAX_STEPS:
                self.game_over = True

        terminated = self.game_over or self.game_won
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement and Aiming
        if movement == 1: # Up
            self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: # Aim Left
            self.player_aim_direction = -1
        elif movement == 4: # Aim Right
            self.player_aim_direction = 1

        self.player_pos.y = np.clip(self.player_pos.y, self.HEIGHT / 2, self.HEIGHT - self.PLAYER_HEIGHT - 30)

        # Shooting
        if space_held and not self.is_reloading and self.shot_cooldown <= 0:
            if self.player_ammo > 0:
                self.player_ammo -= 1
                self.shot_cooldown = self.PLAYER_SHOT_COOLDOWN
                bullet_start_pos = self.player_pos + pygame.Vector2(self.PLAYER_WIDTH / 2, self.PLAYER_HEIGHT / 2)
                bullet_velocity = pygame.Vector2(self.player_aim_direction * 20, 0)
                self.bullets.append({'pos': bullet_start_pos, 'vel': bullet_velocity, 'trail': []})
                # Sound effect placeholder: # pew!
                self._create_particles(bullet_start_pos + pygame.Vector2(self.player_aim_direction * 15, 0), 10, self.COLOR_MUZZLE_FLASH, 5, 4)
            else:
                reward -= 0.1 # Penalty for trying to shoot with no ammo
                # Sound effect placeholder: # empty clip click

        # Reloading
        if shift_held and not self.is_reloading and self.player_ammo < self.PLAYER_MAX_AMMO:
            self.is_reloading = True
            self.reload_timer = self.PLAYER_RELOAD_TIME
            # Sound effect placeholder: # click-clack reload

        return reward

    def _update_game_state(self):
        reward = 0

        # Update timers
        if self.shot_cooldown > 0: self.shot_cooldown -= 1
        if self.is_reloading:
            self.reload_timer -= 1
            if self.reload_timer <= 0:
                self.is_reloading = False
                self.player_ammo = self.PLAYER_MAX_AMMO
                # Sound effect placeholder: # reload complete
        
        # Update bullets
        bullets_to_remove = []
        for i, bullet in enumerate(self.bullets):
            bullet['pos'] += bullet['vel']
            bullet['trail'].append(pygame.Vector2(bullet['pos']))
            if len(bullet['trail']) > 5: bullet['trail'].pop(0)
            
            if not (0 < bullet['pos'].x < self.WIDTH):
                bullets_to_remove.append(i)
                reward -= 0.01 # Penalty for missing

        # Update zombies
        zombies_to_remove = []
        for i, zombie in enumerate(self.zombies):
            zombie['hit_timer'] = max(0, zombie['hit_timer'] - 1)
            
            # Movement
            dir_to_player = (self.player_pos - zombie['pos']).normalize() if (self.player_pos - zombie['pos']).length() > 0 else pygame.Vector2(0,0)
            zombie['pos'] += dir_to_player * zombie['speed']
            
            # Collision with player
            player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
            zombie_rect = pygame.Rect(zombie['pos'].x, zombie['pos'].y, zombie['size'][0], zombie['size'][1])
            if player_rect.colliderect(zombie_rect) and zombie['hit_timer'] == 0:
                self.player_health -= self.ZOMBIE_DAMAGE
                reward -= 1.0
                zombie['hit_timer'] = self.ZOMBIE_HIT_COOLDOWN
                # Sound effect placeholder: # player hurt
                self._create_particles(self.player_pos + pygame.Vector2(self.PLAYER_WIDTH/2, self.PLAYER_HEIGHT/2), 20, self.COLOR_HEALTH, 10, 2)

            # Collision with bullets
            hit_by_bullet = False
            for j, bullet in enumerate(self.bullets):
                if j not in bullets_to_remove and zombie_rect.collidepoint(bullet['pos']):
                    zombie['health'] -= 50
                    zombie['is_hit'] = 5 # Flash for 5 frames
                    reward += 0.1
                    if j not in bullets_to_remove: bullets_to_remove.append(j)
                    self._create_particles(bullet['pos'], 15, self.COLOR_BLOOD, 8, 3)
                    # Sound effect placeholder: # squish
                    hit_by_bullet = True
                    break
            
            if zombie['health'] <= 0:
                self.score += 10
                reward += 1.0
                if i not in zombies_to_remove: zombies_to_remove.append(i)
                self._create_particles(zombie['pos'] + pygame.Vector2(zombie['size'][0]/2, zombie['size'][1]/2), 40, self.COLOR_ZOMBIE, 20, 2)
                # Sound effect placeholder: # zombie death
        
        # Remove dead/off-screen entities
        for i in sorted(bullets_to_remove, reverse=True): del self.bullets[i]
        for i in sorted(zombies_to_remove, reverse=True): del self.zombies[i]

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= p['shrink_rate']

        # Wave progression
        if not self.zombies and not self.game_over and not self.game_won:
            if self.wave_transition_timer == 0:
                self.wave_transition_timer = 90 # 3 seconds
                reward += 100
                # Sound effect placeholder: # wave clear
            
            self.wave_transition_timer -= 1
            if self.wave_transition_timer <= 0:
                self.wave_number += 1
                if self.wave_number <= 5:
                    self._spawn_wave()

        return reward

    def _spawn_wave(self):
        num_zombies = 3 + (self.wave_number - 1) * 2
        zombie_speed = 0.8 + (self.wave_number - 1) * 0.1
        for _ in range(num_zombies):
            side = random.choice([-1, 1])
            start_x = -50 if side == -1 else self.WIDTH + 50
            start_y = random.randint(int(self.HEIGHT/2), self.HEIGHT - 80)
            self.zombies.append({
                'pos': pygame.Vector2(start_x, start_y),
                'health': self.ZOMBIE_MAX_HEALTH,
                'speed': zombie_speed * random.uniform(0.8, 1.2),
                'size': (25, 50),
                'hit_timer': 0,
                'is_hit': 0,
                'bob_offset': random.uniform(0, math.pi * 2)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background
        self._render_background()
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - 30, self.WIDTH, 30))

        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Zombies
        for z in self.zombies:
            color = self.COLOR_ZOMBIE_HIT if z['is_hit'] > 0 else self.COLOR_ZOMBIE
            if z['is_hit'] > 0: z['is_hit'] -= 1
            z_bob = math.sin(self.steps * 0.15 + z['bob_offset']) * 3
            z_rect = pygame.Rect(int(z['pos'].x), int(z['pos'].y + z_bob), z['size'][0], z['size'][1])
            pygame.draw.rect(self.screen, color, z_rect)

        # Player
        player_y = self.player_pos.y + self.bob_offset
        player_rect = pygame.Rect(int(self.player_pos.x), int(player_y), self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Player Gun
        gun_y = player_y + self.PLAYER_HEIGHT / 2 - 2
        if self.player_aim_direction == 1:
            gun_rect = pygame.Rect(int(self.player_pos.x + self.PLAYER_WIDTH/2), int(gun_y), 15, 4)
        else:
            gun_rect = pygame.Rect(int(self.player_pos.x + self.PLAYER_WIDTH/2 - 15), int(gun_y), 15, 4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GUN, gun_rect)
        
        # Bullets
        for b in self.bullets:
            # Trail
            if len(b['trail']) > 1:
                # Use a surface with per-pixel alpha for smooth trails
                trail_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.aalines(trail_surf, self.COLOR_BULLET, False, [(p.x, p.y) for p in b['trail']], 2)
                self.screen.blit(trail_surf, (0, 0))
            # Bullet tip
            pygame.gfxdraw.filled_circle(self.screen, int(b['pos'].x), int(b['pos'].y), 3, self.COLOR_BULLET)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, 200 * health_ratio, 20))
        health_text = self.font_small.render("HEALTH", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Ammo Bar
        ammo_ratio = max(0, self.player_ammo / self.PLAYER_MAX_AMMO)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 35, 150, 15))
        pygame.draw.rect(self.screen, self.COLOR_AMMO, (10, 35, 150 * ammo_ratio, 15))
        ammo_text = self.font_small.render(f"{self.player_ammo}/{self.PLAYER_MAX_AMMO}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (15, 35))

        # Wave and Score
        wave_text = self.font_small.render(f"WAVE: {min(self.wave_number, 5)}/5", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 35))

        # Game State Text
        if self.is_reloading:
            reload_text = self.font_large.render("RELOADING", True, self.COLOR_AMMO)
            text_rect = reload_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 50))
            self.screen.blit(reload_text, text_rect)
        elif self.wave_transition_timer > 0 and not self.game_won:
            wave_clear_text = self.font_large.render(f"WAVE {self.wave_number-1} CLEARED", True, self.COLOR_PLAYER)
            text_rect = wave_clear_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(wave_clear_text, text_rect)
        elif self.game_over:
            end_text = self.font_large.render("GAME OVER", True, self.COLOR_HEALTH)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
        elif self.game_won:
            win_text = self.font_large.render("YOU SURVIVED!", True, self.COLOR_PLAYER)
            text_rect = win_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(win_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "health": self.player_health,
            "ammo": self.player_ammo,
        }

    def _generate_background(self):
        self.background_buildings = []
        for i in range(2): # Two layers for parallax illusion
            layer = []
            for _ in range(20):
                w = random.randint(30, 80)
                h = random.randint(50, 250)
                x = random.randint(-self.WIDTH, self.WIDTH * 2)
                color_val = 30 + i*10 + random.randint(-5, 5)
                color = (color_val, color_val, color_val + 10)
                layer.append({'rect': pygame.Rect(x, self.HEIGHT - h - 30, w, h), 'color': color})
            self.background_buildings.append(layer)

    def _render_background(self):
        for layer in self.background_buildings:
            for building in layer:
                pygame.draw.rect(self.screen, building['color'], building['rect'])

    def _create_particles(self, pos, count, color, max_life, max_speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifespan': random.randint(max_life // 2, max_life),
                'radius': random.uniform(2, 5),
                'color': color,
                'shrink_rate': 0.1
            })

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be used by the evaluation script, which imports the GameEnv class.
    os.environ.pop("SDL_VIDEODRIVER", None) # Allow display for direct play
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1

        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    pygame.quit()