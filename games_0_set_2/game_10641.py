import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:45:14.518081
# Source Brief: brief_00641.md
# Brief Index: 641
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Synthwave Showdown: A top-down shooter Gymnasium environment with a retro 80s aesthetic.
    The player navigates a neon city, fights patrolling enemies, and uses resources
    to activate a temporary shield. The goal is to survive as long as possible.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive waves of enemies in a retro-futuristic city. Collect resources to power your shield "
        "and blast your way to a high score in this top-down synthwave shooter."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to shoot and shift to activate your shield."
    )
    auto_advance = True

    # --- CONSTANTS ---
    # Game world
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    FPS = 30

    # Colors (Synthwave Palette)
    COLOR_BG = (13, 1, 33)
    COLOR_GRID = (48, 25, 52)
    COLOR_BUILDING = (102, 3, 102)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 128, 255)
    COLOR_ENEMY = (255, 0, 128)
    COLOR_ENEMY_GLOW = (128, 0, 128)
    COLOR_PROJECTILE = (255, 255, 0)
    COLOR_RESOURCE = (255, 255, 0)
    COLOR_HEALTH = (0, 255, 0)
    COLOR_SHIELD = (200, 200, 255)
    COLOR_TEXT = (255, 255, 255)
    COLOR_UI_BG = (25, 9, 57, 180) # RGBA

    # Player
    PLAYER_SPEED = 5
    PLAYER_RADIUS = 12
    PLAYER_MAX_HEALTH = 100
    PLAYER_SHOOT_COOLDOWN = 10 # steps
    PLAYER_SHIELD_DURATION = 90 # steps (3 seconds at 30fps)
    PLAYER_SHIELD_COST = 5

    # Enemy
    ENEMY_RADIUS = 10
    ENEMY_HEALTH = 3
    ENEMY_CONTACT_DAMAGE = 10
    
    # Projectile
    PROJECTILE_SPEED = 10
    PROJECTILE_RADIUS = 4
    PROJECTILE_DAMAGE = 1

    # Resource
    RESOURCE_RADIUS = 6
    RESOURCE_SPAWN_INTERVAL = 150 # steps
    MAX_RESOURCES = 5

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_aug = pygame.font.SysFont("monospace", 14, bold=True)

        # --- Game State Initialization ---
        # These are reset in self.reset()
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = None
        self.player_health = None
        self.player_resources = None
        self.player_last_move_dir = None
        self.player_shoot_cooldown_timer = None
        self.player_shield_timer = None

        self.enemies = []
        self.projectiles = []
        self.resources = []
        self.particles = []
        
        self.max_enemies = 0
        self.enemy_spawn_timer = 0
        self.base_enemy_speed = 0.0
        self.resource_spawn_timer = 0

        self.background_surface = self._create_background()
        
        # This is not a standard Gym call, but we'll leave it as it was in the original code.
        # self.reset()
        
        # --- Critical Self-Check ---
        # self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_resources = 0
        self.player_last_move_dir = np.array([0, -1]) # Default to up
        self.player_shoot_cooldown_timer = 0
        self.player_shield_timer = 0

        self.enemies.clear()
        self.projectiles.clear()
        self.resources.clear()
        self.particles.clear()

        # Progression variables
        self.max_enemies = 1
        self.enemy_spawn_timer = 100 # Initial delay
        self.base_enemy_speed = 0.5
        self.resource_spawn_timer = self.RESOURCE_SPAWN_INTERVAL

        # Regenerate background for variety
        self.background_surface = self._create_background()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01 # Small survival reward per step

        # --- Update Cooldowns and Timers ---
        self._update_timers()

        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_player(action[0])
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Handle Collisions and Interactions ---
        reward += self._handle_collisions()

        # --- Handle Spawning ---
        self._handle_spawning()

        # --- Update Score & Steps ---
        self.steps += 1
        self.score += reward

        # --- Check Termination ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False # This environment does not truncate based on time limit in the same way as some others
        if self.steps >= self.MAX_STEPS:
            terminated = True
            truncated = True # Indicate that termination is due to a time limit

        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward = -100.0
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
            # sfx: player_death_explosion
        elif terminated and not self.game_over: # Survived
            self.game_over = True
            reward += 100.0
            # sfx: victory_fanfare

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Private Helper Methods: Update Logic ---

    def _update_timers(self):
        if self.player_shoot_cooldown_timer > 0:
            self.player_shoot_cooldown_timer -= 1
        if self.player_shield_timer > 0:
            self.player_shield_timer -= 1
        if self.enemy_spawn_timer > 0:
            self.enemy_spawn_timer -= 1
        if self.resource_spawn_timer > 0:
            self.resource_spawn_timer -= 1

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Shooting
        if space_held and self.player_shoot_cooldown_timer == 0:
            self._shoot()
        
        # Augmentation (Shield)
        if shift_held and self.player_shield_timer == 0 and self.player_resources >= self.PLAYER_SHIELD_COST:
            self._activate_shield()
            
    def _update_player(self, movement_action):
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement_action == 1: move_vec[1] = -1 # Up
        elif movement_action == 2: move_vec[1] = 1 # Down
        elif movement_action == 3: move_vec[0] = -1 # Left
        elif movement_action == 4: move_vec[0] = 1 # Right

        if np.linalg.norm(move_vec) > 0:
            self.player_last_move_dir = move_vec / np.linalg.norm(move_vec)
            self.player_pos += self.player_last_move_dir * self.PLAYER_SPEED
            
            # Thrust particles
            if self.steps % 2 == 0:
                p_pos = self.player_pos - self.player_last_move_dir * self.PLAYER_RADIUS
                p_vel = -self.player_last_move_dir * random.uniform(1, 2) + self.np_random.random(size=2) * 0.5 - 0.25
                self.particles.append({'pos': p_pos, 'vel': p_vel, 'radius': self.np_random.integers(2, 5), 'color': self.COLOR_PLAYER_GLOW, 'lifetime': 10})


        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _update_projectiles(self):
        self.projectiles = [p for p in self.projectiles if 0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT]
        for p in self.projectiles:
            p['pos'] += p['vel']

    def _update_enemies(self):
        for e in self.enemies:
            # Simple AI: move towards player
            direction = self.player_pos - e['pos']
            dist = np.linalg.norm(direction)
            if dist > 1: # Avoid division by zero and jittering
                e['pos'] += (direction / dist) * self.base_enemy_speed
            
            # Glow pulse
            e['glow_t'] = (e['glow_t'] + 0.1) % (2 * math.pi)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] *= 0.95

    def _handle_collisions(self):
        reward = 0

        # Projectiles vs Enemies
        for proj in self.projectiles[:]:
            for enemy in self.enemies[:]:
                dist = np.linalg.norm(proj['pos'] - enemy['pos'])
                if dist < self.PROJECTILE_RADIUS + self.ENEMY_RADIUS:
                    enemy['health'] -= self.PROJECTILE_DAMAGE
                    self._create_explosion(proj['pos'], self.COLOR_ENEMY, 5) # Hit spark
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    # sfx: enemy_hit
                    if enemy['health'] <= 0:
                        reward += 5
                        self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 25)
                        if enemy in self.enemies: self.enemies.remove(enemy)
                        # sfx: enemy_explosion
                    break

        # Player vs Enemies
        if self.player_shield_timer == 0:
            for enemy in self.enemies:
                dist = np.linalg.norm(self.player_pos - enemy['pos'])
                if dist < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                    self.player_health -= self.ENEMY_CONTACT_DAMAGE
                    self.player_health = max(0, self.player_health)
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER, 15)
                    # sfx: player_damage
                    # Simple knockback
                    direction = self.player_pos - enemy['pos']
                    self.player_pos += (direction / np.linalg.norm(direction)) * 15
                    break # Only take damage from one enemy per frame

        # Player vs Resources
        for res in self.resources[:]:
            dist = np.linalg.norm(self.player_pos - res['pos'])
            if dist < self.PLAYER_RADIUS + self.RESOURCE_RADIUS:
                reward += 1
                self.player_resources += 1
                self.resources.remove(res)
                # sfx: resource_pickup
        
        return reward

    def _handle_spawning(self):
        # Progression: Increase difficulty over time
        if self.steps > 0 and self.steps % 500 == 0:
            self.max_enemies = min(10, self.max_enemies + 1)
        if self.steps > 0 and self.steps % 1000 == 0:
            self.base_enemy_speed = min(2.5, self.base_enemy_speed + 0.1)

        # Spawn Enemies
        if self.enemy_spawn_timer <= 0 and len(self.enemies) < self.max_enemies:
            self._spawn_enemy()
            self.enemy_spawn_timer = self.np_random.integers(150, 251)

        # Spawn Resources
        if self.resource_spawn_timer <= 0 and len(self.resources) < self.MAX_RESOURCES:
            self._spawn_resource()
            self.resource_spawn_timer = self.RESOURCE_SPAWN_INTERVAL

    # --- Private Helper Methods: Actions and Spawning ---
    
    def _shoot(self):
        self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
        proj_pos = self.player_pos + self.player_last_move_dir * (self.PLAYER_RADIUS + 5)
        proj_vel = self.player_last_move_dir * self.PROJECTILE_SPEED
        self.projectiles.append({'pos': proj_pos, 'vel': proj_vel})
        # sfx: player_shoot
        # Muzzle flash
        self.particles.append({'pos': proj_pos.copy(), 'vel': np.array([0,0]), 'radius': 8, 'color': self.COLOR_PROJECTILE, 'lifetime': 4})


    def _activate_shield(self):
        self.player_resources -= self.PLAYER_SHIELD_COST
        self.player_shield_timer = self.PLAYER_SHIELD_DURATION
        # sfx: shield_activate
        # Shield activation effect
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * self.np_random.uniform(1, 3)
            self.particles.append({'pos': self.player_pos.copy(), 'vel': vel, 'radius': self.np_random.integers(2, 5), 'color': self.COLOR_SHIELD, 'lifetime': 20})

    def _spawn_enemy(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ENEMY_RADIUS])
        elif edge == 1: # Bottom
            pos = np.array([self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ENEMY_RADIUS])
        elif edge == 2: # Left
            pos = np.array([-self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        else: # Right
            pos = np.array([self.SCREEN_WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.SCREEN_HEIGHT)])
        
        self.enemies.append({'pos': pos, 'health': self.ENEMY_HEALTH, 'glow_t': self.np_random.uniform(0, 2*math.pi)})

    def _spawn_resource(self):
        pos = np.array([
            self.np_random.uniform(50, self.SCREEN_WIDTH - 50),
            self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
        ])
        self.resources.append({'pos': pos})

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.integers(3, 7),
                'color': color,
                'lifetime': self.np_random.integers(15, 31)
            })

    # --- Private Helper Methods: Rendering ---

    def _get_observation(self):
        # Clear screen with background
        self.screen.blit(self.background_surface, (0, 0))
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _create_background(self):
        bg = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        bg.fill(self.COLOR_BG)
        # Grid lines
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(bg, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(bg, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
        # Random buildings
        if hasattr(self, 'np_random'):
            for _ in range(10):
                w, h = self.np_random.integers(50, 151), self.np_random.integers(50, 151)
                x, y = self.np_random.integers(0, self.SCREEN_WIDTH-w+1), self.np_random.integers(0, self.SCREEN_HEIGHT-h+1)
                pygame.gfxdraw.rectangle(bg, (x, y, w, h), self.COLOR_BUILDING)
        return bg

    def _render_game(self):
        # Particles (drawn first, behind other objects)
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / 30.0))
            self._draw_glow_circle(self.screen, p['pos'], max(0, p['radius']), p['color'], alpha)

        # Resources
        for res in self.resources:
            pos_int = res['pos'].astype(int)
            self._draw_glow_circle(self.screen, pos_int, self.RESOURCE_RADIUS, self.COLOR_RESOURCE)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.RESOURCE_RADIUS-2, self.COLOR_RESOURCE)

        # Projectiles
        for p in self.projectiles:
            pos_int = p['pos'].astype(int)
            self._draw_glow_circle(self.screen, pos_int, self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, self.COLOR_PROJECTILE)

        # Enemies
        for e in self.enemies:
            pos_int = e['pos'].astype(int)
            glow_radius = self.ENEMY_RADIUS + 5 + 3 * math.sin(e['glow_t'])
            self._draw_glow_circle(self.screen, pos_int, glow_radius, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)

        # Player
        if self.player_health > 0:
            pos_int = self.player_pos.astype(int)
            # Shield effect
            if self.player_shield_timer > 0:
                shield_alpha = 50 + 50 * math.sin(self.steps * 0.5)
                shield_radius = self.PLAYER_RADIUS + 8 + 2 * math.sin(self.steps * 0.5)
                self._draw_glow_circle(self.screen, pos_int, shield_radius, self.COLOR_SHIELD, shield_alpha)
            
            self._draw_glow_circle(self.screen, pos_int, self.PLAYER_RADIUS + 8, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Health Bar
        health_percent = self.player_health / self.PLAYER_MAX_HEALTH
        health_bar_width = int(150 * health_percent)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (10, 10, 160, 30))
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (15, 15, 150, 20))
        if health_bar_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH, (15, 15, health_bar_width, 20))
        health_text = self.font_ui.render(f"HP", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (180, 16))

        # Resource Count
        res_text = self.font_ui.render(f"RESOURCES: {self.player_resources}", True, self.COLOR_TEXT)
        text_rect = res_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 16))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, text_rect.inflate(10, 10))
        self.screen.blit(res_text, text_rect)
        
        # Augmentation Status
        aug_surf = pygame.Surface((200, 40), pygame.SRCALPHA)
        aug_surf.fill(self.COLOR_UI_BG)
        
        if self.player_shield_timer > 0:
            status_text = "SHIELD ACTIVE"
            status_color = self.COLOR_SHIELD
        elif self.player_resources >= self.PLAYER_SHIELD_COST:
            status_text = f"SHIELD READY [{self.PLAYER_SHIELD_COST}R]"
            status_color = self.COLOR_PLAYER
        else:
            status_text = f"SHIELD [{self.PLAYER_SHIELD_COST}R]"
            status_color = (100, 100, 100)
        
        aug_text = self.font_aug.render(status_text, True, status_color)
        aug_rect = aug_text.get_rect(center=(100, 20))
        aug_surf.blit(aug_text, aug_rect)
        self.screen.blit(aug_surf, (self.SCREEN_WIDTH/2 - 100, self.SCREEN_HEIGHT - 50))


    def _draw_glow_circle(self, surface, pos, radius, color, alpha_base=100):
        pos = pos.astype(int)
        radius = int(radius)
        if radius <= 0: return
        
        for i in range(radius, 0, -2):
            alpha = int(alpha_base * (1 - (i / radius)**2))
            if alpha > 0:
                pygame.gfxdraw.aacircle(surface, pos[0], pos[1], i, (*color, alpha))
    
    # --- Gymnasium Interface Compliance ---

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "resources": self.player_resources,
            "enemies": len(self.enemies)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
    
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Synthwave Showdown")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Input ---
        movement = 0 # None
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over!")
    print(f"Final Score: {total_reward:.2f}")
    print(f"Info: {info}")
    
    env.close()