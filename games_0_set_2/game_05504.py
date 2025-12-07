
# Generated: 2025-08-28T05:13:55.664277
# Source Brief: brief_05504.md
# Brief Index: 5504

        
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
        "Controls: Arrow keys for isometric movement. Press space to shoot. Hold shift to reload."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of procedurally generated zombies in an isometric arena by strategically shooting and reloading."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Game Feel
    PLAYER_SPEED = 4.0
    ZOMBIE_SPEED = 1.0
    PROJECTILE_SPEED = 15.0
    PLAYER_KNOCKBACK = 15.0
    
    # Sizes
    PLAYER_RADIUS = 12
    ZOMBIE_RADIUS = 10
    PROJECTILE_RADIUS = 4
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_WALL = (60, 60, 70)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 200, 255, 50)
    COLOR_ZOMBIE = (100, 120, 80)
    COLOR_ZOMBIE_GLOW = (150, 50, 50, 60)
    COLOR_PROJECTILE = (255, 200, 0)
    COLOR_MUZZLE_FLASH = (255, 230, 180)
    COLOR_BLOOD = (180, 20, 20)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH = (220, 40, 40)
    COLOR_AMMO = (0, 150, 255)
    COLOR_RELOAD = (255, 165, 0)

    # Game Rules
    MAX_STEPS = 3000
    PLAYER_MAX_HEALTH = 100
    PLAYER_MAX_AMMO = 20
    SHOOT_COOLDOWN = 6 # frames
    RELOAD_TIME = 45 # frames
    ZOMBIE_HEALTH = 3
    ZOMBIE_DAMAGE = 10
    WAVE_CONFIG = [10, 15, 20, 25, 30]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Game state variables are initialized in reset()
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game state
        self.steps = 0
        self.score = 0
        self.reward = 0
        self.game_over = False
        self.victory = False
        self.game_message = ""
        self.message_timer = 0
        
        # Player state
        self.player_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float64)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_aim_dir = np.array([1.0, 0.0]) # Start aiming right
        self.shoot_cooldown = 0
        self.reload_timer = 0
        self.is_reloading = False
        self.prev_space_held = False
        
        # Entities
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        # Wave management
        self.wave_number = 0
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.reward = 0
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        if not self.game_over:
            self._handle_input(movement, space_held, shift_held)
            self._update_player()
            self._update_zombies()
            self._update_projectiles()
            self._update_particles()
            self._handle_collisions()
            self._check_wave_status()
        
        self.steps += 1
        self.score += self.reward
        
        # Update message timer
        if self.message_timer > 0:
            self.message_timer -= 1
        else:
            self.game_message = ""

        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_held, shift_held):
        # Movement and Aiming
        move_vec = np.array([0.0, 0.0])
        if not self.is_reloading:
            if movement == 1: # Up-Left
                move_vec = np.array([-1.0, -1.0])
            elif movement == 2: # Down-Left
                move_vec = np.array([-1.0, 1.0])
            elif movement == 3: # Down-Right
                move_vec = np.array([1.0, 1.0])
            elif movement == 4: # Up-Right
                move_vec = np.array([1.0, -1.0])
        
        if np.linalg.norm(move_vec) > 0:
            self.player_velocity = move_vec / np.linalg.norm(move_vec) * self.PLAYER_SPEED
            self.player_aim_dir = self.player_velocity / self.PLAYER_SPEED
        else:
            self.player_velocity = np.array([0.0, 0.0])

        # Shooting
        shoot_action = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        if shoot_action and self.shoot_cooldown == 0 and self.player_ammo > 0 and not self.is_reloading:
            self._shoot()
        
        # Reloading
        if shift_held and self.player_ammo < self.PLAYER_MAX_AMMO and not self.is_reloading:
            self.is_reloading = True
            self.reload_timer = self.RELOAD_TIME
        
        if self.is_reloading:
            if not shift_held: # Cancel reload
                self.is_reloading = False
                self.reload_timer = 0
            else:
                self.reload_timer -= 1
                if self.reload_timer <= 0:
                    self.player_ammo = self.PLAYER_MAX_AMMO
                    self.is_reloading = False
                    # sfx: reload_complete.wav

    def _shoot(self):
        self.player_ammo -= 1
        self.shoot_cooldown = self.SHOOT_COOLDOWN
        
        proj_pos = self.player_pos.copy()
        proj_vel = self.player_aim_dir * self.PROJECTILE_SPEED
        self.projectiles.append({'pos': proj_pos, 'vel': proj_vel, 'radius': self.PROJECTILE_RADIUS})
        
        # Muzzle flash
        flash_pos = self.player_pos + self.player_aim_dir * (self.PLAYER_RADIUS + 5)
        for _ in range(10):
            angle = random.uniform(-0.5, 0.5) + math.atan2(self.player_aim_dir[1], self.player_aim_dir[0])
            speed = random.uniform(2, 6)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': flash_pos.copy(), 'vel': vel, 'radius': random.randint(2, 5),
                'color': self.COLOR_MUZZLE_FLASH, 'lifespan': 8
            })
        # sfx: shoot.wav
    
    def _update_player(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        
        self.player_pos += self.player_velocity
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.screen_width - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.screen_height - self.PLAYER_RADIUS)

    def _update_zombies(self):
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            dist = np.linalg.norm(direction)
            if dist > 1: # Avoid division by zero and jittering
                z['pos'] += direction / dist * self.ZOMBIE_SPEED

    def _update_projectiles(self):
        new_projectiles = []
        for p in self.projectiles:
            p['pos'] += p['vel']
            if 0 < p['pos'][0] < self.screen_width and 0 < p['pos'][1] < self.screen_height:
                new_projectiles.append(p)
            else:
                self.reward -= 0.01 # Wasted bullet
        self.projectiles = new_projectiles

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] = max(0, p['radius'] * 0.9)
            
    def _handle_collisions(self):
        # Projectiles vs Zombies
        surviving_projectiles = []
        for p in self.projectiles:
            hit = False
            for z in self.zombies:
                if np.linalg.norm(p['pos'] - z['pos']) < p['radius'] + z['radius']:
                    hit = True
                    z['health'] -= 1
                    self.reward += 0.1 # Zombie hit
                    self._create_blood_splatter(p['pos'])
                    # sfx: zombie_hit.wav
                    break
            if not hit:
                surviving_projectiles.append(p)
        self.projectiles = surviving_projectiles
        
        # Check for dead zombies
        surviving_zombies = []
        for z in self.zombies:
            if z['health'] > 0:
                surviving_zombies.append(z)
            else:
                self.reward += 1.0 # Zombie kill
                # sfx: zombie_die.wav
        self.zombies = surviving_zombies

        # Player vs Zombies
        for z in self.zombies:
            if np.linalg.norm(self.player_pos - z['pos']) < self.PLAYER_RADIUS + z['radius']:
                self.player_health -= self.ZOMBIE_DAMAGE
                self.player_health = max(0, self.player_health)
                # sfx: player_hurt.wav
                
                # Knockback
                knockback_dir = self.player_pos - z['pos']
                if np.linalg.norm(knockback_dir) > 0:
                    knockback_dir /= np.linalg.norm(knockback_dir)
                else: # If perfectly overlapped
                    knockback_dir = np.array([1.0, 0.0])
                self.player_pos += knockback_dir * self.PLAYER_KNOCKBACK
                break # Only take damage from one zombie per frame

    def _create_blood_splatter(self, pos):
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'radius': random.randint(1, 4),
                'color': self.COLOR_BLOOD, 'lifespan': random.randint(15, 30)
            })

    def _spawn_wave(self):
        if self.wave_number >= len(self.WAVE_CONFIG):
            return

        num_zombies = self.WAVE_CONFIG[self.wave_number]
        for _ in range(num_zombies):
            # Spawn on screen edges
            side = random.randint(0, 3)
            if side == 0: # Top
                x, y = random.uniform(0, self.screen_width), -self.ZOMBIE_RADIUS
            elif side == 1: # Bottom
                x, y = random.uniform(0, self.screen_width), self.screen_height + self.ZOMBIE_RADIUS
            elif side == 2: # Left
                x, y = -self.ZOMBIE_RADIUS, random.uniform(0, self.screen_height)
            else: # Right
                x, y = self.screen_width + self.ZOMBIE_RADIUS, random.uniform(0, self.screen_height)

            self.zombies.append({
                'pos': np.array([x, y], dtype=np.float64),
                'radius': self.ZOMBIE_RADIUS,
                'health': self.ZOMBIE_HEALTH
            })
        
        self.wave_number += 1
        self.game_message = f"WAVE {self.wave_number}"
        self.message_timer = 60 # 2 seconds at 30fps

    def _check_wave_status(self):
        if len(self.zombies) == 0 and not self.game_over:
            if self.wave_number >= len(self.WAVE_CONFIG):
                self.victory = True
                self.game_over = True
                self.reward += 500
                self.game_message = "YOU WIN!"
                self.message_timer = self.MAX_STEPS
            else:
                self.reward += 100
                self._spawn_wave()

    def _check_termination(self):
        if self.player_health <= 0 and not self.game_over:
            self.game_over = True
            self.reward -= 100
            self.game_message = "GAME OVER"
            self.message_timer = self.MAX_STEPS
            # sfx: game_over.wav
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        return self.game_over

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw arena walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.screen_width, self.screen_height), 10)

        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

        # Draw zombies
        for z in self.zombies:
            pos = (int(z['pos'][0]), int(z['pos'][1]))
            rad = z['radius']
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad + 4, self.COLOR_ZOMBIE_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, self.COLOR_ZOMBIE)

        # Draw projectiles
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            rad = p['radius']
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, self.COLOR_PROJECTILE)

        # Draw player
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        rad = self.PLAYER_RADIUS
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad + 8, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad + 4, self.COLOR_PLAYER_GLOW)
        # Player body
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], rad, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], rad, self.COLOR_PLAYER)
        # Aiming indicator
        aim_end = self.player_pos + self.player_aim_dir * (rad + 2)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, pos, (int(aim_end[0]), int(aim_end[1])), 3)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, (50, 0, 0), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (10, 10, int(200 * health_ratio), 20))
        
        # Ammo Count
        ammo_text = self.font_small.render(f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10, 35))
        
        # Reloading indicator
        if self.is_reloading:
            reload_ratio = 1.0 - (self.reload_timer / self.RELOAD_TIME)
            bar_width = 50
            bar_height = 5
            bar_x = self.player_pos[0] - bar_width / 2
            bar_y = self.player_pos[1] - self.PLAYER_RADIUS - 15
            pygame.draw.rect(self.screen, (80, 80, 80), (int(bar_x), int(bar_y), bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_RELOAD, (int(bar_x), int(bar_y), int(bar_width * reload_ratio), bar_height))

        # Score and Wave
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 10, 35))
        
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{len(self.WAVE_CONFIG)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.screen_width - wave_text.get_width() - 10, 10))
        
        # Game Messages
        if self.message_timer > 0:
            msg_surf = self.font_large.render(self.game_message, True, self.COLOR_UI_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(msg_surf, msg_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "wave": self.wave_number,
            "zombies_remaining": len(self.zombies),
            "victory": self.victory,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    import os
    # Set a specific video driver to run without a physical display
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment")
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Diagonal movement mapping
        up = keys[pygame.K_UP]
        down = keys[pygame.K_DOWN]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]

        if up and left: movement = 1
        elif down and left: movement = 2
        elif down and right: movement = 3
        elif up and right: movement = 4
        # Allow cardinal directions too for better playability
        elif up: movement = 1 # Simplified to UL for this mapping
        elif down: movement = 3 # Simplified to DR
        elif left: movement = 2 # Simplified to DL
        elif right: movement = 4 # Simplified to UR

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Victory: {info['victory']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(30) # Run at 30 FPS
        
    env.close()