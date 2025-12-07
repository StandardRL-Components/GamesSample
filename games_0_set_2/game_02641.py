
# Generated: 2025-08-28T05:29:57.025636
# Source Brief: brief_02641.md
# Brief Index: 2641

        
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
        "Controls: Arrow keys to move. Press space to shoot at the nearest zombie."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a zombie horde in a top-down arena shooter. Eliminate all zombies to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.W, self.H = 640, 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_ARENA = (40, 40, 50)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_FLASH = (255, 255, 255)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_BAR_BG = (128, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (0, 200, 0)
        
        # Game constants
        self.MAX_STEPS = 1000
        self.INITIAL_ZOMBIES = 25
        self.PLAYER_SPEED = 4
        self.PLAYER_SIZE = 16
        self.PLAYER_INITIAL_HEALTH = 100
        self.PLAYER_INITIAL_AMMO = 50
        self.PLAYER_SHOOT_COOLDOWN = 5 # frames
        self.PLAYER_DMG_COOLDOWN = 15 # frames
        self.ZOMBIE_SIZE = 14
        self.ZOMBIE_INITIAL_SPEED = 1.0
        self.ZOMBIE_SPEED_INCREASE = 0.05
        self.PROJECTILE_SPEED = 10
        self.PROJECTILE_SIZE = (2, 8)
        self.ARENA_PADDING = 10

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_shoot_timer = None
        self.player_flash_timer = None
        self.zombies = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.W / 2, self.H / 2)
        self.player_health = self.PLAYER_INITIAL_HEALTH
        self.player_ammo = self.PLAYER_INITIAL_AMMO
        self.player_shoot_timer = 0
        self.player_flash_timer = 0
        
        self.zombies = []
        self._spawn_zombies(self.INITIAL_ZOMBIES)
        
        self.projectiles = [] # {pos: Vector2, vel: Vector2, rect: Rect}
        self.particles = [] # {pos: Vector2, vel: Vector2, lifetime: int, color: tuple, size: int}
        
        return self._get_observation(), self._get_info()

    def _spawn_zombies(self, num_zombies):
        for _ in range(num_zombies):
            while True:
                x = self.np_random.integers(self.ARENA_PADDING, self.W - self.ARENA_PADDING - self.ZOMBIE_SIZE)
                y = self.np_random.integers(self.ARENA_PADDING, self.H - self.ARENA_PADDING - self.ZOMBIE_SIZE)
                new_zombie_rect = pygame.Rect(x, y, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
                
                # Ensure not too close to player
                if pygame.Vector2(new_zombie_rect.center).distance_to(self.player_pos) < 100:
                    continue
                
                # Ensure not overlapping with other zombies
                if not any(new_zombie_rect.colliderect(zombie['rect']) for zombie in self.zombies):
                    self.zombies.append({'rect': new_zombie_rect, 'pos': pygame.Vector2(x, y)})
                    break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward

        self._handle_input_and_cooldowns(action)
        self._update_player(action)
        self._update_zombies()
        reward += self._update_projectiles()
        self._update_particles()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self.steps += 1
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        self.score += reward

        if terminated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input_and_cooldowns(self, action):
        # Handle shooting
        space_held = action[1] == 1
        if space_held and self.player_shoot_timer == 0 and self.player_ammo > 0:
            self._shoot()
            self.player_ammo -= 1
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
            # sfx: player_shoot.wav
        
        # Decrement cooldowns
        if self.player_shoot_timer > 0:
            self.player_shoot_timer -= 1
        if self.player_flash_timer > 0:
            self.player_flash_timer -= 1

    def _update_player(self, action):
        movement = action[0]
        direction = pygame.Vector2(0, 0)
        if movement == 1: direction.y = -1 # Up
        if movement == 2: direction.y = 1  # Down
        if movement == 3: direction.x = -1 # Left
        if movement == 4: direction.x = 1  # Right
        
        if direction.length() > 0:
            direction.normalize_ip()
        
        self.player_pos += direction * self.PLAYER_SPEED
        
        # Clamp player position to arena bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.ARENA_PADDING + self.PLAYER_SIZE/2, self.W - self.ARENA_PADDING - self.PLAYER_SIZE/2)
        self.player_pos.y = np.clip(self.player_pos.y, self.ARENA_PADDING + self.PLAYER_SIZE/2, self.H - self.ARENA_PADDING - self.PLAYER_SIZE/2)

    def _shoot(self):
        nearest_zombie = self._find_nearest_zombie()
        if not nearest_zombie:
            return

        direction = (pygame.Vector2(nearest_zombie['rect'].center) - self.player_pos).normalize()
        
        # Create projectile
        proj_pos = self.player_pos.copy()
        proj_rect = pygame.Rect(proj_pos.x - self.PROJECTILE_SIZE[0]/2, proj_pos.y - self.PROJECTILE_SIZE[1]/2, self.PROJECTILE_SIZE[0], self.PROJECTILE_SIZE[1])
        self.projectiles.append({'pos': proj_pos, 'vel': direction * self.PROJECTILE_SPEED, 'rect': proj_rect, 'angle': direction.angle_to(pygame.Vector2(0, -1))})
        
        # Muzzle flash
        self._create_particles(self.player_pos + direction * 10, self.COLOR_PROJECTILE, 5, 0.5, 3)

    def _find_nearest_zombie(self):
        if not self.zombies:
            return None
        
        closest_zombie = None
        min_dist_sq = float('inf')
        
        for zombie in self.zombies:
            dist_sq = self.player_pos.distance_squared_to(zombie['pos'])
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_zombie = zombie
        return closest_zombie

    def _update_zombies(self):
        zombie_speed = self.ZOMBIE_INITIAL_SPEED + self.ZOMBIE_SPEED_INCREASE * (self.steps // 100)
        for zombie in self.zombies:
            direction = (self.player_pos - zombie['pos'])
            if direction.length() > 0:
                direction.normalize_ip()
            zombie['pos'] += direction * zombie_speed
            zombie['rect'].center = zombie['pos']

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for proj in self.projectiles:
            proj['pos'] += proj['vel']
            proj['rect'].center = proj['pos']
            
            # Check boundaries
            if not (self.ARENA_PADDING < proj['pos'].x < self.W - self.ARENA_PADDING and \
                    self.ARENA_PADDING < proj['pos'].y < self.H - self.ARENA_PADDING):
                reward -= 1 # Penalty for hitting a wall
                # sfx: bullet_ricochet.wav
            else:
                projectiles_to_keep.append(proj)
        self.projectiles = projectiles_to_keep
        return reward
    
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    def _handle_collisions(self):
        reward = 0
        
        # Player vs Zombies
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        
        if self.player_flash_timer == 0: # If not invincible
            for zombie in self.zombies:
                if player_rect.colliderect(zombie['rect']):
                    self.player_health -= 10
                    reward -= 5
                    self.player_flash_timer = self.PLAYER_DMG_COOLDOWN
                    # sfx: player_hit.wav
                    self._create_particles(self.player_pos, self.COLOR_PLAYER_FLASH, 10, 2, 5)
                    break # Only take damage from one zombie per frame
        
        # Projectiles vs Zombies
        projectiles_to_remove = set()
        zombies_to_remove = set()
        
        for i, proj in enumerate(self.projectiles):
            for j, zombie in enumerate(self.zombies):
                if j in zombies_to_remove: continue
                if proj['rect'].colliderect(zombie['rect']):
                    projectiles_to_remove.add(i)
                    zombies_to_remove.add(j)
                    reward += 10
                    # sfx: zombie_die.wav
                    self._create_particles(zombie['pos'], self.COLOR_ZOMBIE, 20, 1.5, 7)
                    break # A projectile can only hit one zombie
        
        if zombies_to_remove:
            self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]
        if projectiles_to_remove:
            self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]
            
        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            # sfx: game_over.wav
            return True, -100 # Player died
        if not self.zombies:
            # sfx: level_win.wav
            return True, 100 # All zombies killed
        if self.steps >= self.MAX_STEPS:
            return True, 0 # Max steps reached
        return False, 0

    def _create_particles(self, pos, color, count, speed_scale, lifetime):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': self.np_random.integers(lifetime, lifetime + 5),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render arena
        arena_rect = pygame.Rect(self.ARENA_PADDING, self.ARENA_PADDING, self.W - 2*self.ARENA_PADDING, self.H - 2*self.ARENA_PADDING)
        pygame.draw.rect(self.screen, self.COLOR_ARENA, arena_rect)
        
        # Render all game elements
        self._render_zombies()
        self._render_projectiles()
        self._render_particles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_zombies(self):
        for zombie in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie['rect'])

    def _render_projectiles(self):
        for proj in self.projectiles:
            # Create a rotated surface for the projectile
            rotated_surf = pygame.transform.rotate(pygame.Surface(self.PROJECTILE_SIZE, pygame.SRCALPHA), proj['angle'])
            rotated_surf.fill(self.COLOR_PROJECTILE)
            rect = rotated_surf.get_rect(center=proj['pos'])
            self.screen.blit(rotated_surf, rect)

    def _render_particles(self):
        for p in self.particles:
            if p['size'] > 1:
                pygame.draw.rect(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y), int(p['size']), int(p['size'])))

    def _render_player(self):
        color = self.COLOR_PLAYER_FLASH if self.player_flash_timer > 0 else self.COLOR_PLAYER
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        pygame.draw.rect(self.screen, color, player_rect)
        pygame.draw.rect(self.screen, tuple(min(255, c*0.7) for c in color), player_rect, 1) # Outline

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_INITIAL_HEALTH)
        bar_w, bar_h = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_w * health_ratio), bar_h))
        
        # Ammo Count
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (self.W - ammo_text.get_width() - 10, 10))
        
        # Zombie Count
        zombie_text = self.font_large.render(f"{len(self.zombies)} ZOMBIES", True, self.COLOR_TEXT)
        self.screen.blit(zombie_text, (self.W/2 - zombie_text.get_width()/2, self.H - zombie_text.get_height() - 5))

        if self.game_over:
            if not self.zombies:
                end_text = "VICTORY"
                end_color = self.COLOR_PLAYER
            else:
                end_text = "GAME OVER"
                end_color = self.COLOR_ZOMBIE
            
            text_surface = self.font_large.render(end_text, True, end_color)
            text_rect = text_surface.get_rect(center=(self.W / 2, self.H / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies_left": len(self.zombies)
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
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage to test the environment visually
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Zombie Arena")
    
    terminated = False
    
    # Use a simple agent to play the game
    # 0=none, 1=up, 2=down, 3=left, 4=right
    # 0=release, 1=hold
    action = [0, 0, 0] # [movement, space, shift]
    
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Shooting
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        # Shift
        if keys[pygame.K_SHIFT]:
            action[2] = 1

        if terminated:
            # Wait for a key press to reset
            if any(keys):
                obs, info = env.reset()
                terminated = False
        else:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Control the frame rate

    env.close()