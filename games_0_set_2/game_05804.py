
# Generated: 2025-08-28T06:08:33.115776
# Source Brief: brief_05804.md
# Brief Index: 5804

        
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
        "Controls: Use ← and → to move, ↑ to jump. Press Space to shoot."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a zombie horde in a side-scrolling shooter. Reach the escape zone on the right before time runs out or your health depletes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and World Dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH = 3000
        self.GROUND_Y = self.SCREEN_HEIGHT - 50
        self.FPS = 60
        self.TIME_LIMIT_SECONDS = 120
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup (headless)
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_score = pygame.font.Font(None, 40)
        
        # Colors
        self.COLOR_BG_DARK = (20, 22, 30)
        self.COLOR_BG_MID = (30, 32, 42)
        self.COLOR_BG_LIGHT = (40, 45, 55)
        self.COLOR_GROUND = (50, 45, 50)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_ZOMBIE = (140, 20, 20)
        self.COLOR_ZOMBIE_HIT = (255, 50, 50)
        self.COLOR_BULLET = (255, 230, 180)
        self.COLOR_ESCAPE_ZONE = (100, 255, 100)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_GOOD = (80, 200, 80)
        self.COLOR_HEALTH_BAD = (200, 80, 80)
        self.COLOR_MUZZLE_FLASH = (255, 220, 150)
        self.COLOR_BLOOD = (180, 0, 0)

        # Game Physics & Parameters
        self.PLAYER_SPEED = 4
        self.PLAYER_JUMP_STRENGTH = -11
        self.GRAVITY = 0.45
        self.BULLET_SPEED = 12
        self.SHOOT_COOLDOWN = 15  # frames

        self.initial_zombie_speed = 1.0
        self.initial_zombie_spawn_rate = 2.0 * self.FPS # every 2 seconds

        # Will be initialized in reset()
        self.player = {}
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.background_layers = []
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.shoot_cooldown_timer = 0
        self.zombie_spawn_timer = 0
        self.zombie_speed = 0
        self.zombie_spawn_rate = 0
        self.damage_flash_timer = 0
        self.camera_shake = 0
        self.reward_buffer = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player State
        self.player = {
            'rect': pygame.Rect(100, self.GROUND_Y - 40, 20, 40),
            'vx': 0,
            'vy': 0,
            'on_ground': False,
            'health': 100,
            'facing': 1  # 1 for right, -1 for left
        }
        
        # Game State
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reward_buffer = 0

        # Timers & Difficulty
        self.shoot_cooldown_timer = 0
        self.zombie_spawn_timer = self.initial_zombie_spawn_rate
        self.zombie_speed = self.initial_zombie_speed
        self.zombie_spawn_rate = self.initial_zombie_spawn_rate
        self.damage_flash_timer = 0
        self.camera_shake = 0

        # Procedural background
        self._generate_background()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_buffer = 0 # Reset per-step reward
        
        # --- Handle Input ---
        self._handle_input(action)

        # --- Update Game Logic ---
        self._update_player()
        self._update_bullets()
        self._update_zombies()
        self._update_particles()
        self._spawn_zombies()
        self._update_game_state()

        # --- Termination and Reward ---
        reward = self._calculate_reward()
        terminated = self._check_termination()
        self.game_over = terminated

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Horizontal Movement
        if movement == 3:  # Left
            self.player['vx'] = -self.PLAYER_SPEED
            self.player['facing'] = -1
        elif movement == 4:  # Right
            self.player['vx'] = self.PLAYER_SPEED
            self.player['facing'] = 1
        else:
            self.player['vx'] = 0

        # Jumping
        if movement == 1 and self.player['on_ground']:
            self.player['vy'] = self.PLAYER_JUMP_STRENGTH
            self.player['on_ground'] = False
            # Sound: Player Jump

        # Shooting
        if space_held and self.shoot_cooldown_timer <= 0:
            self._fire_bullet()
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
            # Sound: Player Shoot

    def _update_player(self):
        # Apply physics
        self.player['vy'] += self.GRAVITY
        self.player['rect'].x += self.player['vx']
        self.player['rect'].y += self.player['vy']

        # World bounds
        self.player['rect'].left = max(0, self.player['rect'].left)
        self.player['rect'].right = min(self.WORLD_WIDTH, self.player['rect'].right)

        # Ground collision
        if self.player['rect'].bottom >= self.GROUND_Y:
            self.player['rect.bottom'] = self.GROUND_Y
            self.player['vy'] = 0
            if not self.player['on_ground']: # Just landed
                # Sound: Player Land
                for _ in range(5):
                    self._create_particle(self.player['rect'].midbottom, self.COLOR_GROUND, count=1, speed_range=(1,3), angle_range=(-160, -20))
            self.player['on_ground'] = True
        
        # Update timers
        if self.shoot_cooldown_timer > 0:
            self.shoot_cooldown_timer -= 1
        if self.damage_flash_timer > 0:
            self.damage_flash_timer -= 1

    def _fire_bullet(self):
        start_pos = self.player['rect'].center
        bullet = {
            'rect': pygame.Rect(start_pos[0], start_pos[1] - 5, 8, 4),
            'vx': self.BULLET_SPEED * self.player['facing']
        }
        self.bullets.append(bullet)
        # Muzzle flash effect
        self._create_particle(start_pos, self.COLOR_MUZZLE_FLASH, count=10, speed_range=(2, 6), angle_range=(-30, 30), lifespan=8, gravity=0.1, direction_offset=self.player['facing'])


    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet['rect'].x += bullet['vx']
            
            # Remove bullets that are off-screen
            bullet_screen_x = bullet['rect'].x - self.camera_x
            if not (0 < bullet_screen_x < self.SCREEN_WIDTH):
                self.bullets.remove(bullet)
                continue

            # Check for collision with zombies
            for zombie in self.zombies[:]:
                if bullet['rect'].colliderect(zombie['rect']):
                    zombie['health'] -= 20
                    zombie['hit_timer'] = 5 # Flash on hit
                    self.bullets.remove(bullet)
                    # Sound: Zombie Hit
                    self._create_particle(zombie['rect'].center, self.COLOR_BLOOD, count=15, speed_range=(1, 4), angle_range=(0, 360), lifespan=20)
                    if zombie['health'] <= 0:
                        self.zombies.remove(zombie)
                        self.score += 1
                        self.reward_buffer += 1
                        # Sound: Zombie Die
                    break

    def _update_zombies(self):
        for zombie in self.zombies:
            # Move towards player
            if self.player['rect'].centerx > zombie['rect'].centerx:
                zombie['rect'].x += self.zombie_speed
            else:
                zombie['rect'].x -= self.zombie_speed
            
            if zombie['hit_timer'] > 0:
                zombie['hit_timer'] -= 1

            # Check for collision with player
            if zombie['rect'].colliderect(self.player['rect']) and self.damage_flash_timer <= 0:
                self.player['health'] -= 10
                self.damage_flash_timer = 30 # 0.5s invulnerability
                self.camera_shake = 10
                self.reward_buffer -= 10 # Small penalty for getting hit
                # Sound: Player Hurt

    def _update_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += p['gravity']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            side = random.choice([-1, 1])
            spawn_x = self.camera_x + (self.SCREEN_WIDTH + 50) if side == 1 else self.camera_x - 50
            
            zombie = {
                'rect': pygame.Rect(spawn_x, self.GROUND_Y - 45, 25, 45),
                'health': 100,
                'hit_timer': 0
            }
            self.zombies.append(zombie)
            self.zombie_spawn_timer = self.zombie_spawn_rate

    def _update_game_state(self):
        # Update camera to follow player
        self.camera_x = self.player['rect'].centerx - self.SCREEN_WIDTH / 2
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))
        
        if self.camera_shake > 0:
            self.camera_shake -= 1

        # Difficulty scaling
        time_factor = self.steps / self.FPS
        self.zombie_speed = self.initial_zombie_speed + time_factor * 0.02
        self.zombie_spawn_rate = max(0.4 * self.FPS, self.initial_zombie_spawn_rate - time_factor * 0.5)

    def _calculate_reward(self):
        # Add a small reward for moving right (progressing)
        if self.player['vx'] > 0:
            self.reward_buffer += 0.001
        
        # Check terminal conditions for large rewards/penalties
        if self.player['health'] <= 0:
            return -100
        
        if self.player['rect'].right >= self.WORLD_WIDTH:
            return 100
        
        return self.reward_buffer

    def _check_termination(self):
        # Win: reach escape zone
        if self.player['rect'].right >= self.WORLD_WIDTH:
            self.score += 100 # Bonus for winning
            return True
        # Lose: health <= 0
        if self.player['health'] <= 0:
            return True
        # Lose: time out
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        # --- Rendering Pipeline ---
        self.screen.fill(self.COLOR_BG_DARK)
        
        # Apply camera shake
        shake_offset_x = random.randint(-self.camera_shake, self.camera_shake) if self.camera_shake > 0 else 0
        shake_offset_y = random.randint(-self.camera_shake, self.camera_shake) if self.camera_shake > 0 else 0
        effective_camera_x = self.camera_x + shake_offset_x
        
        self._render_background(effective_camera_x, shake_offset_y)
        self._render_game_elements(effective_camera_x, shake_offset_y)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, cam_x, cam_y):
        # Parallax background layers
        for i, layer in enumerate(self.background_layers):
            parallax_factor = 0.1 + 0.2 * i
            layer_cam_x = cam_x * parallax_factor
            for rect, color in layer:
                # Use modulo to tile the background infinitely
                screen_x = rect.x - layer_cam_x
                screen_x_wrapped = screen_x % self.WORLD_WIDTH - rect.width
                
                # Draw multiple instances for seamless wrapping
                for offset in [-self.WORLD_WIDTH, 0, self.WORLD_WIDTH]:
                     pygame.draw.rect(self.screen, color, (screen_x_wrapped + offset, rect.y + cam_y, rect.width, rect.height))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y + cam_y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))
    
    def _render_game_elements(self, cam_x, cam_y):
        # Escape Zone
        escape_rect = pygame.Rect(self.WORLD_WIDTH - 20, 0, 20, self.GROUND_Y)
        screen_escape_x = escape_rect.x - cam_x
        if screen_escape_x < self.SCREEN_WIDTH:
            alpha = 100 + math.sin(self.steps * 0.1) * 30
            pygame.gfxdraw.box(self.screen, (int(screen_escape_x), int(cam_y), escape_rect.width, escape_rect.height), (*self.COLOR_ESCAPE_ZONE, int(alpha)))
            for i in range(5):
                line_alpha = 150 + math.sin(self.steps * 0.2 + i) * 50
                pygame.draw.line(self.screen, (*self.COLOR_ESCAPE_ZONE, line_alpha), (screen_escape_x + i*4, cam_y), (screen_escape_x + i*4, self.GROUND_Y + cam_y), 2)

        # Zombies
        for z in self.zombies:
            color = self.COLOR_ZOMBIE_HIT if z['hit_timer'] > 0 else self.COLOR_ZOMBIE
            screen_rect = z['rect'].move(-cam_x, cam_y)
            pygame.draw.rect(self.screen, color, screen_rect)

        # Player
        player_color = self.COLOR_PLAYER
        if self.damage_flash_timer > 0 and self.steps % 4 < 2:
             player_color = self.COLOR_ZOMBIE_HIT
        screen_player_rect = self.player['rect'].move(-cam_x, cam_y)
        pygame.draw.rect(self.screen, player_color, screen_player_rect)
        
        # Bullets
        for b in self.bullets:
            screen_rect = b['rect'].move(-cam_x, cam_y)
            pygame.draw.rect(self.screen, self.COLOR_BULLET, screen_rect)

        # Particles
        for p in self.particles:
            size = max(1, p['lifespan'] / p['max_lifespan'] * p['size'])
            pygame.draw.circle(self.screen, p['color'], (int(p['x'] - cam_x), int(p['y'] + cam_y)), int(size))

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player['health'] / 100)
        health_color = self.COLOR_HEALTH_GOOD if health_ratio > 0.3 else self.COLOR_HEALTH_BAD
        bar_width = 200
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, bar_width * health_ratio, 20))

        # Timer
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        timer_text = f"TIME: {int(time_left // 60):02}:{int(time_left % 60):02}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (self.SCREEN_WIDTH - timer_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_score.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, ((self.SCREEN_WIDTH - score_surf.get_width()) / 2, self.SCREEN_HEIGHT - 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player['health'],
            "time_left": max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS)),
            "player_pos": (self.player['rect'].x, self.player['rect'].y),
        }

    def _generate_background(self):
        self.background_layers = []
        colors = [self.COLOR_BG_LIGHT, self.COLOR_BG_MID]
        for i, color in enumerate(colors):
            layer = []
            for _ in range(30): # Number of buildings per layer
                w = random.randint(50, 200)
                h = random.randint(50, 250 - i*50)
                x = random.randint(0, self.WORLD_WIDTH)
                y = self.GROUND_Y - h
                layer.append((pygame.Rect(x, y, w, h), color))
            self.background_layers.append(layer)

    def _create_particle(self, pos, color, count, speed_range, angle_range, lifespan=15, size=3, gravity=0.2, direction_offset=1):
        for _ in range(count):
            angle = math.radians(random.uniform(*angle_range))
            speed = random.uniform(*speed_range)
            p = {
                'x': pos[0],
                'y': pos[1],
                'vx': math.cos(angle) * speed * direction_offset,
                'vy': math.sin(angle) * speed,
                'lifespan': random.randint(lifespan // 2, lifespan),
                'max_lifespan': lifespan,
                'color': color,
                'size': size,
                'gravity': gravity
            }
            self.particles.append(p)
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
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
    env = GameEnv()
    obs, info = env.reset()
    
    # Override auto_advance for human play
    env.auto_advance = False 
    
    running = True
    total_reward = 0
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False
    }
    
    # We need a screen to display the game for human play
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Escape")

    while running:
        # Construct the action from held keys
        move_action = 0 # none
        if keys_held[pygame.K_UP]: move_action = 1
        elif keys_held[pygame.K_DOWN]: move_action = 2
        elif keys_held[pygame.K_LEFT]: move_action = 3
        elif keys_held[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys_held[pygame.K_SPACE] else 0
        shift_action = 1 if keys_held[pygame.K_LSHIFT] else 0
        
        action = [move_action, space_action, shift_action]
        
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Control the frame rate for human play
        env.clock.tick(env.FPS)

    pygame.quit()