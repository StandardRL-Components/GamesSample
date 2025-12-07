import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings adapted to the implemented game
    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to auto-seek ammo. Press Space to shoot."
    )
    game_description = (
        "Survive for 5 minutes against an ever-growing horde of zombies in this top-down arcade shooter."
    )

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
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_GAMEOVER = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (15, 15, 15)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_OUTLINE = (0, 200, 100)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_ZOMBIE_OUTLINE = (180, 20, 20)
        self.COLOR_AMMO = (255, 220, 0)
        self.COLOR_BULLET = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)
        self.COLOR_HEALTH_BAR = (0, 220, 0)
        self.COLOR_HIT_FLASH = (255, 0, 0)
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_last_move_dir = None
        self.zombies = []
        self.bullets = []
        self.ammo_drops = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.hit_flash_timer = 0
        self.shoot_cooldown = 0
        self.zombie_spawn_timer = 0
        self.ammo_spawn_timer = 0
        self.rng = None
        self.zombie_size = None # Will be initialized in reset
        
        # Call reset to populate initial state
        # self.reset() is called by the wrapper, but we need to init for validation
        # We will properly seed and reset again when the user calls reset()
        self.reset(seed=0)

        # Validate implementation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            # If no seed and no existing generator, create a new one
            self.rng = np.random.default_rng()

        # Player state
        self.player_pos = pygame.math.Vector2(self.W / 2, self.H / 2)
        self.player_health = 50
        self.MAX_HEALTH = 50
        self.player_ammo = 50
        self.MAX_AMMO = 100
        self.player_size = 20
        self.player_speed = 4.0
        self.zombie_size = 18
        self.player_last_move_dir = pygame.math.Vector2(0, -1)

        # Entity lists
        self.zombies = []
        self.bullets = []
        self.ammo_drops = []
        self.particles = []

        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = 3000

        # Timers and cooldowns
        self.hit_flash_timer = 0
        self.shoot_cooldown = 0
        self.zombie_spawn_timer = 0
        self.ammo_spawn_timer = self.rng.integers(150, 300) # 15-30s

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # -- 1. Handle Input & Player Movement --
        reward += self._handle_input(action)
        self.player_pos[0] %= self.W
        self.player_pos[1] %= self.H

        # -- 2. Update Game Logic --
        reward += self._update_bullets()
        self._update_zombies()
        self._update_particles()
        
        # -- 3. Spawning --
        self._spawn_zombies()
        self._spawn_ammo_drops()

        # -- 4. Handle Collisions --
        reward += self._handle_collisions()

        # -- 5. Check Termination --
        terminated = self.player_health <= 0 or self.steps >= self.max_steps
        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 100 # Penalty for dying
            else: # Survived
                reward += 100 # Bonus for winning

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        move_vec = pygame.math.Vector2(0, 0)
        
        # Shift action overrides manual movement
        if shift_held and self.ammo_drops:
            nearest_ammo = min(self.ammo_drops, key=lambda a: self.player_pos.distance_to(a['pos']))
            direction = (nearest_ammo['pos'] - self.player_pos)
            if direction.length() > 0:
                move_vec = direction.normalize()
        else:
            if movement == 1: move_vec.y = -1 # Up
            elif movement == 2: move_vec.y = 1 # Down
            elif movement == 3: move_vec.x = -1 # Left
            elif movement == 4: move_vec.x = 1 # Right

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.player_speed
            self.player_last_move_dir = move_vec.copy()
        
        # Handle shooting
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        if space_held and self.shoot_cooldown == 0 and self.player_ammo > 0:
            self.player_ammo -= 1
            reward -= 0.01 # Penalty for using ammo
            self.shoot_cooldown = 3 # 3 steps cooldown
            
            # Find nearest zombie to target
            target_dir = self.player_last_move_dir
            if self.zombies:
                nearest_zombie = min(self.zombies, key=lambda z: self.player_pos.distance_to(z['pos']))
                direction = (nearest_zombie['pos'] - self.player_pos)
                if direction.length() > 0:
                    target_dir = direction.normalize()

            # Create bullet
            # sound: shoot.wav
            bullet_pos = self.player_pos + target_dir * (self.player_size / 2)
            self.bullets.append({'pos': bullet_pos, 'vel': target_dir * 15.0})
            
            # Muzzle flash
            for _ in range(10):
                flash_vel = target_dir * self.rng.uniform(2, 5) + pygame.math.Vector2(self.rng.uniform(-4, 4), self.rng.uniform(-4, 4))
                self.particles.append({
                    'pos': self.player_pos.copy(), 'vel': flash_vel,
                    'lifespan': self.rng.integers(4, 8), 'size': self.rng.uniform(2, 4),
                    'color': (255, 255, 150)
                })
        return reward

    def _update_zombies(self):
        # Difficulty scaling: speed increases by 0.01 per second (10 steps)
        current_zombie_speed = 1.0 + (self.steps / 10) * 0.01

        for z in self.zombies:
            direction = (self.player_pos - z['pos'])
            if direction.length() > 0:
                direction.normalize_ip()
            z['pos'] += direction * current_zombie_speed
            z['pos'].x %= self.W
            z['pos'].y %= self.H
    
    def _update_bullets(self):
        reward = 0
        bullets_to_keep = []
        for b in self.bullets:
            b['pos'] += b['vel']
            
            hit_zombie = False
            # Use a copy for iteration to allow removing from original list
            for z in list(self.zombies):
                if b['pos'].distance_to(z['pos']) < (self.zombie_size / 2):
                    self.zombies.remove(z)
                    hit_zombie = True
                    self.score += 10
                    reward += 0.1 # Reward for killing a zombie
                    # sound: zombie_die.wav
                    # Death particles
                    for _ in range(20):
                        vel = pygame.math.Vector2(self.rng.uniform(-3, 3), self.rng.uniform(-3, 3))
                        self.particles.append({
                            'pos': z['pos'].copy(), 'vel': vel,
                            'lifespan': self.rng.integers(10, 20), 'size': self.rng.uniform(1, 3),
                            'color': self.COLOR_ZOMBIE
                        })
                    break # Bullet can only hit one zombie
            
            if not hit_zombie and 0 <= b['pos'].x < self.W and 0 <= b['pos'].y < self.H:
                bullets_to_keep.append(b)
        
        self.bullets = bullets_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] * 0.95)

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            # Rate increases by 0.001 per second (10 steps)
            current_spawn_rate_per_sec = 0.5 + (self.steps / 10) * 0.001
            steps_per_spawn = max(5, 10 / current_spawn_rate_per_sec) # 10 steps per second
            self.zombie_spawn_timer = steps_per_spawn

            # Spawn on edge of screen
            edge = self.rng.choice(['top', 'bottom', 'left', 'right'])
            pos = pygame.math.Vector2(0, 0)
            if edge == 'top': pos.x, pos.y = self.rng.uniform(0, self.W), -self.zombie_size
            elif edge == 'bottom': pos.x, pos.y = self.rng.uniform(0, self.W), self.H + self.zombie_size
            elif edge == 'left': pos.x, pos.y = -self.zombie_size, self.rng.uniform(0, self.H)
            elif edge == 'right': pos.x, pos.y = self.W + self.zombie_size, self.rng.uniform(0, self.H)
            
            self.zombies.append({'pos': pos, 'size': self.zombie_size})

    def _spawn_ammo_drops(self):
        self.ammo_spawn_timer -= 1
        if self.ammo_spawn_timer <= 0 and len(self.ammo_drops) < 5:
            self.ammo_spawn_timer = self.rng.integers(150, 300) # 15-30s
            pos = pygame.math.Vector2(self.rng.uniform(50, self.W - 50), self.rng.uniform(50, self.H - 50))
            self.ammo_drops.append({'pos': pos, 'size': 12})

    def _handle_collisions(self):
        reward = 0
        
        # Player-Zombie collision
        zombies_to_keep = []
        for z in self.zombies:
            if self.player_pos.distance_to(z['pos']) < (self.player_size / 2 + z['size'] / 2):
                self.player_health -= 10
                self.hit_flash_timer = 5 # Flash for 5 frames
                # sound: player_hurt.wav
            else:
                zombies_to_keep.append(z)
        self.zombies = zombies_to_keep

        # Player-Ammo collision
        ammo_to_keep = []
        for a in self.ammo_drops:
            if self.player_pos.distance_to(a['pos']) < (self.player_size / 2 + a['size'] / 2):
                self.player_ammo = min(self.MAX_AMMO, self.player_ammo + 20)
                reward += 1 # Reward for collecting ammo
                # sound: ammo_pickup.wav
            else:
                ammo_to_keep.append(a)
        self.ammo_drops = ammo_to_keep

        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render ammo drops (blinking)
        blink_scale = 0.8 + 0.2 * math.sin(self.steps * 0.3)
        for a in self.ammo_drops:
            size = int(a['size'] * blink_scale)
            if size > 0:
                pos_x, pos_y = int(a['pos'].x), int(a['pos'].y)
                pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, size, self.COLOR_AMMO)
                pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, size, self.COLOR_AMMO)

        # Render zombies
        for z in self.zombies:
            rect = pygame.Rect(z['pos'].x - z['size']/2, z['pos'].y - z['size']/2, z['size'], z['size'])
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, rect)
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE_OUTLINE, rect, 2)

        # Render player
        player_rect = pygame.Rect(self.player_pos.x - self.player_size/2, self.player_pos.y - self.player_size/2, self.player_size, self.player_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect, 2)

        # Render bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, self.COLOR_BULLET, (int(b['pos'].x), int(b['pos'].y)), 2)
        
        # Render particles
        for p in self.particles:
            if p['size'] < 1: continue
            size_int = int(p['size'])
            alpha = int(255 * (p['lifespan'] / 20)) # Fade out
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((size_int*2, size_int*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size_int, size_int), size_int)
            self.screen.blit(temp_surf, (int(p['pos'].x - size_int), int(p['pos'].y - size_int)))

        # Render hit flash
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1
            flash_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            alpha = int(100 * (self.hit_flash_timer / 5))
            flash_surf.fill((*self.COLOR_HIT_FLASH, alpha))
            self.screen.blit(flash_surf, (0, 0))

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_width * health_ratio, 20))

        # Timer
        time_text = f"SURVIVE: {max(0, self.max_steps - self.steps)}"
        time_surf = self.FONT_UI.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.W - time_surf.get_width() - 10, 10))

        # Ammo count
        ammo_text = f"AMMO: {self.player_ammo}"
        ammo_surf = self.FONT_UI.render(ammo_text, True, self.COLOR_TEXT)
        self.screen.blit(ammo_surf, (self.W/2 - ammo_surf.get_width()/2, self.H - 30))

        # Game Over message
        if self.game_over:
            if self.player_health <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU SURVIVED!"
            
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            go_surf = self.FONT_GAMEOVER.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(go_surf, (self.W/2 - go_surf.get_width()/2, self.H/2 - go_surf.get_height()/2))
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "zombies": len(self.zombies)
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
        obs, info = self.reset(seed=42)
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    # Un-comment the next line to run with display
    os.environ["SDL_VIDEODRIVER"] = "x11"
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
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
        total_reward += reward
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Game Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- Resetting ---")
                obs, info = env.reset()
                total_reward = 0

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            running = False # Or wait for reset key

        clock.tick(30) # Limit to 30 FPS for human playability
        
    env.close()