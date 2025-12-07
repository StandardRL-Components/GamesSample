import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set headless mode for pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot in your last moved direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a relentless zombie horde for 60 seconds in this top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        
        # Fonts - use a common monospace font
        try:
            self.font_small = pygame.font.SysFont("Consolas", 18)
            self.font_large = pygame.font.SysFont("Consolas", 30)
        except pygame.error:
            self.font_small = pygame.font.SysFont(None, 24)
            self.font_large = pygame.font.SysFont(None, 40)

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_AMMO = (255, 220, 0)
        self.COLOR_BULLET = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 80)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_PARTICLE = (255, 80, 80)

        # Game constants
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 4.0
        self.ZOMBIE_SIZE = 16
        self.ZOMBIE_SPEED = 1.0
        self.AMMO_PACK_SIZE = 12
        self.BULLET_SIZE = 4
        self.BULLET_SPEED = 12.0
        self.MAX_STEPS = 1800  # 60 seconds at 30 FPS
        
        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.player_rect = None
        self.player_health = 0
        self.player_ammo = 0
        self.last_move_direction = None
        self.zombies = []
        self.bullets = []
        self.ammo_packs = []
        self.particles = []
        self.muzzle_flash = None
        
        self.zombie_spawn_timer = 0
        self.zombie_spawn_rate = 20
        self.ammo_spawn_timer = 0
        self.shoot_cooldown = 0
        self.shoot_cooldown_max = 6 # ~5 shots per second

        # Initialize state. This also sets up self.np_random.
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_rect = pygame.Rect(self.WIDTH // 2 - self.PLAYER_SIZE // 2, 
                                       self.HEIGHT // 2 - self.PLAYER_SIZE // 2, 
                                       self.PLAYER_SIZE, self.PLAYER_SIZE)
        self.player_health = 100
        self.player_ammo = 50
        self.last_move_direction = np.array([0, -1.0]) # Start aiming up

        self.zombies = []
        self.bullets = []
        self.ammo_packs = []
        self.particles = []
        self.muzzle_flash = None

        self.zombie_spawn_timer = 0
        self.zombie_spawn_rate = 20 # Initial spawn rate
        self.ammo_spawn_timer = 0
        self.shoot_cooldown = 0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held is unused per brief
        
        # Update timers and cooldowns
        self._update_timers()
        
        # Handle player actions
        self._handle_input(movement, space_held)

        # Update game objects
        self._update_bullets()
        self._update_zombies()
        self._update_particles()

        # Handle spawns
        self._spawn_zombies()
        self._spawn_ammo_packs()

        # Handle collisions and calculate event-based rewards
        reward += self._handle_collisions()
        
        # Continuous survival reward
        reward += 0.1
        
        self.steps += 1

        # Update difficulty
        if self.steps > 0 and self.steps % 300 == 0 and self.zombie_spawn_rate > 10:
            self.zombie_spawn_rate -= 1

        terminated = self._check_termination()
        
        if terminated:
            if self.win:
                reward += 100 # Goal-oriented reward
            else: # Died
                reward = -100 # Terminal penalty

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_timers(self):
        self.zombie_spawn_timer += 1
        self.ammo_spawn_timer += 1
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.muzzle_flash and self.muzzle_flash['lifespan'] > 0:
            self.muzzle_flash['lifespan'] -= 1
        else:
            self.muzzle_flash = None

    def _handle_input(self, movement, space_held):
        # Movement
        move_vector = np.array([0.0, 0.0])
        if movement == 1: # Up
            move_vector[1] = -1.0
        elif movement == 2: # Down
            move_vector[1] = 1.0
        elif movement == 3: # Left
            move_vector[0] = -1.0
        elif movement == 4: # Right
            move_vector[0] = 1.0
        
        if np.any(move_vector):
            # Normalize for consistent diagonal speed
            norm = np.linalg.norm(move_vector)
            if norm > 0:
                move_vector /= norm
                self.last_move_direction = move_vector.copy()

        self.player_rect.x += move_vector[0] * self.PLAYER_SPEED
        self.player_rect.y += move_vector[1] * self.PLAYER_SPEED

        # Boundary checks
        self.player_rect.left = max(0, self.player_rect.left)
        self.player_rect.right = min(self.WIDTH, self.player_rect.right)
        self.player_rect.top = max(0, self.player_rect.top)
        self.player_rect.bottom = min(self.HEIGHT, self.player_rect.bottom)

        # Shooting
        if space_held and self.player_ammo > 0 and self.shoot_cooldown == 0:
            # sfx: player_shoot.wav
            self.player_ammo -= 1
            self.shoot_cooldown = self.shoot_cooldown_max
            
            bullet_start_pos = self.player_rect.center
            bullet = {'rect': pygame.Rect(bullet_start_pos[0] - self.BULLET_SIZE//2, 
                                          bullet_start_pos[1] - self.BULLET_SIZE//2, 
                                          self.BULLET_SIZE, self.BULLET_SIZE),
                      'vel': self.last_move_direction * self.BULLET_SPEED}
            self.bullets.append(bullet)
            
            # Muzzle flash effect
            self.muzzle_flash = {'pos': bullet_start_pos, 'radius': 10, 'lifespan': 2}

    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet['rect'].x += bullet['vel'][0]
            bullet['rect'].y += bullet['vel'][1]
            if not self.screen.get_rect().colliderect(bullet['rect']):
                self.bullets.remove(bullet)

    def _update_zombies(self):
        for zombie in self.zombies:
            dx = self.player_rect.centerx - zombie['rect'].centerx
            dy = self.player_rect.centery - zombie['rect'].centery
            dist = math.hypot(dx, dy)
            if dist > 0:
                dx, dy = dx / dist, dy / dist
            zombie['rect'].x += dx * self.ZOMBIE_SPEED
            zombie['rect'].y += dy * self.ZOMBIE_SPEED
            
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_zombies(self):
        if self.zombie_spawn_timer >= self.zombie_spawn_rate:
            self.zombie_spawn_timer = 0
            # sfx: zombie_spawn.wav
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                x = self.np_random.integers(self.WIDTH)
                y = -self.ZOMBIE_SIZE
            elif edge == 1: # Bottom
                x = self.np_random.integers(self.WIDTH)
                y = self.HEIGHT
            elif edge == 2: # Left
                x = -self.ZOMBIE_SIZE
                y = self.np_random.integers(self.HEIGHT)
            else: # Right
                x = self.WIDTH
                y = self.np_random.integers(self.HEIGHT)
            
            zombie_rect = pygame.Rect(x, y, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            self.zombies.append({'rect': zombie_rect})

    def _spawn_ammo_packs(self):
        if self.ammo_spawn_timer >= 100: # Spawn every 100 frames
            self.ammo_spawn_timer = 0
            if len(self.ammo_packs) < 5: # Limit max ammo packs on screen
                # sfx: ammo_spawn.wav
                pos_x = self.np_random.integers(50, self.WIDTH - 50)
                pos_y = self.np_random.integers(50, self.HEIGHT - 50)
                pack_rect = pygame.Rect(pos_x, pos_y, self.AMMO_PACK_SIZE, self.AMMO_PACK_SIZE)
                self.ammo_packs.append(pack_rect)

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Zombies
        for bullet in self.bullets[:]:
            # FIX: Use collidelist to get the index of the FIRST zombie hit.
            # collidelistall returns a list of indices, which caused the TypeError.
            collided_zombie_index = bullet['rect'].collidelist([z['rect'] for z in self.zombies])
            if collided_zombie_index != -1:
                # sfx: zombie_die.wav
                self.bullets.remove(bullet)
                zombie_to_remove = self.zombies.pop(collided_zombie_index)
                self.score += 10
                reward += 1.0
                self._create_death_particles(zombie_to_remove['rect'].center)
                break # The bullet is consumed, so we stop checking this bullet

        # Player vs Zombies
        zombie_hit_index = self.player_rect.collidelist([z['rect'] for z in self.zombies])
        if zombie_hit_index != -1:
            # sfx: player_hit.wav
            self.player_health -= 10
            zombie_to_remove = self.zombies.pop(zombie_hit_index) # Zombie is destroyed on contact
            self._create_death_particles(zombie_to_remove['rect'].center, (200,100,100))
        
        self.player_health = max(0, self.player_health)

        # Player vs Ammo Packs
        pack_hit_index = self.player_rect.collidelist(self.ammo_packs)
        if pack_hit_index != -1:
            # sfx: ammo_pickup.wav
            self.ammo_packs.pop(pack_hit_index)
            self.player_ammo += 20
            self.score += 5
            reward += 0.5
                
        return reward

    def _create_death_particles(self, pos, color=None):
        if color is None:
            color = self.COLOR_PARTICLE
        for _ in range(self.np_random.integers(10, 20)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(10, 25)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            self.win = False
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = True
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        # pygame.surfarray.array3d returns (width, height, 3)
        # We need (height, width, 3), so we transpose
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render ammo packs
        for pack in self.ammo_packs:
            pygame.draw.rect(self.screen, self.COLOR_AMMO, pack)
            pygame.gfxdraw.rectangle(self.screen, pack, (*self.COLOR_AMMO, 150))

        # Render zombies
        for zombie in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, zombie['rect'])

        # Render particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / 20.0))))
            # This requires a surface with per-pixel alpha to look good, but we'll fake it
            # For a simple surface, we just draw with the solid color
            size = max(1, int(p['lifespan'] / 5))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        # Render player
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect)
        # Aiming indicator
        aim_end_x = self.player_rect.centerx + self.last_move_direction[0] * (self.PLAYER_SIZE)
        aim_end_y = self.player_rect.centery + self.last_move_direction[1] * (self.PLAYER_SIZE)
        pygame.draw.line(self.screen, (255,255,255), self.player_rect.center, (int(aim_end_x), int(aim_end_y)), 2)

        # Render bullets
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, bullet['rect'])
            
        # Render muzzle flash
        if self.muzzle_flash:
            alpha = 255 * (self.muzzle_flash['lifespan'] / 2.0)
            radius = int(self.muzzle_flash['radius'] * (self.muzzle_flash['lifespan'] / 2.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(self.muzzle_flash['pos'][0]), int(self.muzzle_flash['pos'][1]), radius, (*self.COLOR_BULLET, int(alpha)))

    def _render_ui(self):
        # Health Bar
        health_bar_width = 200
        health_ratio = self.player_health / 100.0
        current_health_width = int(health_bar_width * health_ratio)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, 20))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, current_health_width, 20))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / 30.0)
        timer_text = f"{time_left:.1f}"
        timer_surf = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (self.WIDTH // 2 - timer_surf.get_width() // 2, 5))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Ammo
        ammo_text = f"AMMO: {self.player_ammo}"
        ammo_surf = self.font_small.render(ammo_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_surf, (self.WIDTH - ammo_surf.get_width() - 10, 30))
        
        # Game Over / Win Text
        if self.game_over:
            message = "YOU SURVIVED!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_surf = self.font_large.render(message, True, color)
            self.screen.blit(end_surf, (self.WIDTH//2 - end_surf.get_width()//2, self.HEIGHT//2 - end_surf.get_height()//2))

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
        
# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Check if a display is available for human play
    try:
        # Re-enable video driver for display
        os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'macOS' etc.
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Zombie Survival")
        human_play = True
    except pygame.error:
        print("No display available. Skipping human play example.")
        human_play = False

    if human_play:
        clock = pygame.time.Clock()
        running = True
        total_reward = 0
        
        # Game loop
        while running:
            # Action mapping for human play
            keys = pygame.key.get_pressed()
            
            movement = 0 # No-op
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Wait for a moment before auto-resetting or quitting
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0
            
            clock.tick(30) # Run at 30 FPS
            
        env.close()