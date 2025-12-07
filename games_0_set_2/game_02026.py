
# Generated: 2025-08-27T19:01:47.961485
# Source Brief: brief_02026.md
# Brief Index: 2026

        
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
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in this side-scrolling shooter. "
        "Hold your ground and aim carefully to last as long as you can."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WORLD_WIDTH = self.SCREEN_WIDTH * 2

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_CITY_NEAR = (25, 30, 50)
        self.COLOR_CITY_FAR = (20, 24, 42)
        self.COLOR_GROUND = (48, 38, 35)
        self.COLOR_PLAYER = (50, 220, 250)
        self.COLOR_PLAYER_DMG = (255, 100, 100)
        self.COLOR_ZOMBIE = (110, 130, 110)
        self.COLOR_ZOMBIE_DMG = (255, 255, 255)
        self.COLOR_PROJECTILE = (100, 255, 180)
        self.COLOR_BLOOD = (180, 20, 20)
        self.COLOR_MUZZLE_FLASH = (255, 230, 150)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BG = (40, 45, 70, 180)
        self.COLOR_HEALTH_BAR = (100, 220, 120)
        self.COLOR_HEALTH_BAR_BG = (200, 80, 80)

        # Fonts
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 64)
        
        # Game constants
        self.MAX_STEPS = 1800
        self.GROUND_Y = self.SCREEN_HEIGHT - 60
        self.PLAYER_SPEED = 6
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_SHOOT_COOLDOWN = 6
        self.PROJECTILE_SPEED = 15
        self.PROJECTILE_DAMAGE = 5
        self.ZOMBIE_MAX_HEALTH = 20
        self.ZOMBIE_DAMAGE = 10
        self.ZOMBIE_MAX_COUNT = 20
        
        # Initialize state variables
        self.np_random = None
        self.player_rect = None
        self.player_health = 0
        self.player_facing_direction = 1
        self.player_damage_timer = 0
        self.shoot_cooldown = 0
        self.last_space_held = False
        self.muzzle_flash_timer = 0
        self.camera_offset_x = 0
        self.camera_shake_timer = 0
        self.zombies = []
        self.projectiles = []
        self.particles = []
        self.city_scape_far = []
        self.city_scape_near = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_this_step = 0
        self.zombie_spawn_timer = 0
        self.zombie_spawn_rate = 20
        self.zombie_speed = 1.0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        player_w, player_h = 20, 40
        self.player_rect = pygame.Rect(
            self.WORLD_WIDTH // 2 - player_w // 2, 
            self.GROUND_Y - player_h, 
            player_w, 
            player_h
        )
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_direction = 1
        self.player_damage_timer = 0
        self.shoot_cooldown = 0
        self.last_space_held = False
        self.muzzle_flash_timer = 0
        self.camera_shake_timer = 0
        
        self.zombies.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.zombie_spawn_timer = 0
        self.zombie_spawn_rate = 20
        self.zombie_speed = 1.0

        if not self.city_scape_far:
            self._generate_cityscape()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.reward_this_step = 0
        
        self._handle_input(action)
        self._update_player()
        self._update_zombies()
        self._update_projectiles()
        self._handle_collisions()
        self._update_difficulty()
        self._update_particles()
        self._update_camera()
        
        self.steps += 1
        terminated = self._check_termination()
        
        reward = self._calculate_reward(terminated)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Movement
        if movement == 3: # Left
            self.player_rect.x -= self.PLAYER_SPEED
            self.player_facing_direction = -1
        elif movement == 4: # Right
            self.player_rect.x += self.PLAYER_SPEED
            self.player_facing_direction = 1
            
        # Shooting
        if space_held and not self.last_space_held and self.shoot_cooldown == 0:
            self._fire_projectile()
            self.shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN
            self.muzzle_flash_timer = 2
            # sfx: player_shoot.wav
        
        self.last_space_held = space_held

    def _update_player(self):
        self.player_rect.x = max(0, min(self.player_rect.x, self.WORLD_WIDTH - self.player_rect.width))
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.player_damage_timer > 0: self.player_damage_timer -= 1
        if self.muzzle_flash_timer > 0: self.muzzle_flash_timer -= 1
        if self.camera_shake_timer > 0: self.camera_shake_timer -= 1

    def _update_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0 and len(self.zombies) < self.ZOMBIE_MAX_COUNT:
            self._spawn_zombie()
            self.zombie_spawn_timer = self.zombie_spawn_rate

        for z in self.zombies:
            if z['rect'].x < self.player_rect.centerx:
                z['rect'].x += self.zombie_speed
            else:
                z['rect'].x -= self.zombie_speed
            
            z['damage_timer'] = max(0, z['damage_timer'] - 1)
            z['bob'] = math.sin(self.steps * 0.2 + z['rect'].x * 0.1) * 2

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['rect'].x += p['dir'] * self.PROJECTILE_SPEED
            if not (0 <= p['rect'].x < self.WORLD_WIDTH):
                self.projectiles.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _update_difficulty(self):
        if self.steps == 600:
            self.zombie_spawn_rate = 15
        elif self.steps == 1200:
            self.zombie_spawn_rate = 10
        
        if self.steps > 0 and self.steps % 300 == 0:
            self.zombie_speed = min(3.0, self.zombie_speed + 0.2)

    def _handle_collisions(self):
        # Projectiles vs Zombies
        for p in self.projectiles[:]:
            for z in self.zombies[:]:
                if p['rect'].colliderect(z['rect']):
                    z['health'] -= self.PROJECTILE_DAMAGE
                    z['damage_timer'] = 5
                    self.reward_this_step += 1.0
                    self.score += 10
                    self._create_particles(z['rect'].center, self.COLOR_BLOOD, 10)
                    self.projectiles.remove(p)
                    # sfx: zombie_hit.wav
                    if z['health'] <= 0:
                        self.zombies.remove(z)
                        self.score += 50
                        self._create_particles(z['rect'].center, self.COLOR_BLOOD, 30)
                        # sfx: zombie_die.wav
                    break
        
        # Zombies vs Player
        if self.player_damage_timer == 0:
            for z in self.zombies:
                if self.player_rect.colliderect(z['rect']):
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.player_damage_timer = 30 # Invincibility frames
                    self.camera_shake_timer = 8
                    self.reward_this_step -= 0.5
                    self._create_particles(self.player_rect.center, self.COLOR_PLAYER_DMG, 5)
                    # sfx: player_hurt.wav
                    if self.player_health <= 0:
                        self.player_health = 0
                        self.game_over = True
                        # sfx: game_over.wav
                    break

    def _update_camera(self):
        target_cam_x = self.player_rect.centerx - self.SCREEN_WIDTH / 2
        self.camera_offset_x += (target_cam_x - self.camera_offset_x) * 0.1
        self.camera_offset_x = max(0, min(self.camera_offset_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))
        if self.camera_shake_timer > 0:
            self.camera_offset_x += self.np_random.uniform(-5, 5)
            
    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _calculate_reward(self, terminated):
        reward = self.reward_this_step + 0.1 # Survival reward
        if terminated and self.player_health > 0:
            reward += 100.0
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Background
        for building in self.city_scape_far:
            b_rect = building.move(-self.camera_offset_x * 0.2, 0)
            pygame.draw.rect(self.screen, self.COLOR_CITY_FAR, b_rect)
        for building in self.city_scape_near:
            b_rect = building.move(-self.camera_offset_x * 0.5, 0)
            pygame.draw.rect(self.screen, self.COLOR_CITY_NEAR, b_rect)
            
        # Ground
        ground_rect = pygame.Rect(0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)

        # Entities (sorted by y for potential future layering)
        entities = []
        entities.extend(self.zombies)
        entities.append({'rect': self.player_rect, 'type': 'player'})
        
        for entity in entities:
            if 'type' in entity and entity['type'] == 'player':
                self._render_player()
            else:
                self._render_zombie(entity)

        # Projectiles
        for p in self.projectiles:
            p_rect = p['rect'].move(-self.camera_offset_x, 0)
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, p_rect)
            pygame.gfxdraw.filled_circle(self.screen, p_rect.centerx, p_rect.centery, p_rect.height//2 + 2, (*self.COLOR_PROJECTILE, 80))

        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0] - self.camera_offset_x), int(p['pos'][1]))
            alpha = max(0, min(255, int(p['lifespan'] * (255 / p['max_lifespan']))))
            color = (*p['color'], alpha)
            size = max(1, int(p['size'] * (p['lifespan'] / p['max_lifespan'])))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

    def _render_player(self):
        if self.player_damage_timer > 0 and self.steps % 2 == 0:
            return # Flicker when damaged
        
        p_rect = self.player_rect.move(-self.camera_offset_x, 0)
        color = self.COLOR_PLAYER if self.player_damage_timer == 0 else self.COLOR_PLAYER_DMG
        pygame.draw.rect(self.screen, color, p_rect, border_radius=3)
        
        # Gun
        gun_y = p_rect.centery
        gun_x_offset = self.player_rect.width // 2
        gun_x = p_rect.x + gun_x_offset if self.player_facing_direction == 1 else p_rect.right - gun_x_offset
        gun_end_x = gun_x + 12 * self.player_facing_direction
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (gun_x, gun_y), (gun_end_x, gun_y), 4)

        # Muzzle Flash
        if self.muzzle_flash_timer > 0:
            flash_pos = (gun_end_x + 5 * self.player_facing_direction, gun_y)
            radius = self.np_random.integers(8, 12)
            pygame.gfxdraw.filled_circle(self.screen, flash_pos[0], flash_pos[1], radius, (*self.COLOR_MUZZLE_FLASH, 200))
            pygame.gfxdraw.filled_circle(self.screen, flash_pos[0], flash_pos[1], radius // 2, (255, 255, 255, 220))

    def _render_zombie(self, zombie):
        z_rect = zombie['rect'].move(-self.camera_offset_x, zombie['bob'])
        color = self.COLOR_ZOMBIE if zombie['damage_timer'] == 0 else self.COLOR_ZOMBIE_DMG
        pygame.draw.rect(self.screen, color, z_rect, border_radius=3)
        
    def _render_ui(self):
        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        bg_rect = pygame.Rect(10, 10, bar_width, bar_height)
        fg_rect = pygame.Rect(10, 10, int(bar_width * health_pct), bar_height)
        
        ui_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(ui_surf, self.COLOR_UI_BG, ui_surf.get_rect(), border_radius=4)
        pygame.draw.rect(ui_surf, self.COLOR_HEALTH_BAR_BG, ui_surf.get_rect(), border_radius=4)
        if health_pct > 0:
            pygame.draw.rect(ui_surf, self.COLOR_HEALTH_BAR, (0, 0, fg_rect.width, fg_rect.height), border_radius=4)
        self.screen.blit(ui_surf, bg_rect.topleft)

        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 12))

        # Timer
        time_left = self.MAX_STEPS - self.steps
        time_text = self.font_main.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH // 2 - time_text.get_width() // 2, 12))
        
        # Game Over / Win Text
        if self.game_over:
            if self.player_health > 0:
                end_text = self.font_large.render("YOU SURVIVED", True, self.COLOR_HEALTH_BAR)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_HEALTH_BAR_BG)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "zombies": len(self.zombies),
        }

    def _fire_projectile(self):
        start_pos_y = self.player_rect.centery
        start_pos_x = self.player_rect.right if self.player_facing_direction == 1 else self.player_rect.left
        proj_rect = pygame.Rect(start_pos_x, start_pos_y - 2, 10, 4)
        self.projectiles.append({'rect': proj_rect, 'dir': self.player_facing_direction})

    def _spawn_zombie(self):
        w, h = 22, 44
        side = self.np_random.choice([-1, 1])
        if side == -1: # Left
            x = self.camera_offset_x - w
        else: # Right
            x = self.camera_offset_x + self.SCREEN_WIDTH + w
        
        x = max(0, min(x, self.WORLD_WIDTH - w))

        zombie = {
            'rect': pygame.Rect(x, self.GROUND_Y - h, w, h),
            'health': self.ZOMBIE_MAX_HEALTH,
            'damage_timer': 0,
            'bob': 0
        }
        self.zombies.append(zombie)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _generate_cityscape(self):
        for i in range(20): # Far
            w = self.np_random.integers(40, 120)
            h = self.np_random.integers(50, 200)
            x = self.np_random.integers(-self.SCREEN_WIDTH, self.WORLD_WIDTH + self.SCREEN_WIDTH)
            self.city_scape_far.append(pygame.Rect(x, self.GROUND_Y - h, w, h))
        for i in range(15): # Near
            w = self.np_random.integers(60, 150)
            h = self.np_random.integers(80, 250)
            x = self.np_random.integers(-self.SCREEN_WIDTH, self.WORLD_WIDTH + self.SCREEN_WIDTH)
            self.city_scape_near.append(pygame.Rect(x, self.GROUND_Y - h, w, h))

    def close(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a window to display the game
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Player controls
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause before reset
            obs, info = env.reset()
            total_reward = 0
        
        env.clock.tick(60) # Control the frame rate for human playability
        
    env.close()