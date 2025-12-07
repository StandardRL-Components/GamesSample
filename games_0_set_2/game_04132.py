import os
import math
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import os
import pygame


# Set headless mode for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to jump, ←→ to move. Press space to attack."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling action game. Jump and attack monsters to survive and achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH = 640
        self.HEIGHT = 400

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
        self.game_font = pygame.font.Font(None, 36)
        self.combo_font = pygame.font.Font(None, 28)
        
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GROUND = (40, 42, 54)
        self.COLOR_PLAYER = (80, 250, 123)
        self.COLOR_PLAYER_ATTACK = (255, 255, 255)
        self.COLOR_MONSTER_1 = (255, 85, 85)
        self.COLOR_MONSTER_2 = (255, 121, 198)
        self.COLOR_TEXT = (248, 248, 242)
        self.COLOR_HIT_FLASH = (255, 249, 138)
        self.COLOR_SCREEN_FLASH = (150, 0, 0)
        
        # Parallax background layers
        self.parallax_layers = [
            {"speed": 0.1, "y": 200, "size": (40, 80), "color": (25, 28, 42), "stars": []},
            {"speed": 0.2, "y": 150, "size": (2, 2), "color": (100, 100, 120), "stars": []}
        ]

        # Game constants
        self.GRAVITY = 0.6
        self.GROUND_Y = self.HEIGHT - 50
        self.MAX_STEPS = 2000
        self.WIN_CONDITION = 20

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_on_ground = None
        self.player_facing_right = None
        self.player_attack_timer = None
        self.player_attack_cooldown = None
        self.player_invincibility_timer = None
        self.lives = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.monsters = None
        self.particles = None
        self.camera_x = None
        self.monsters_defeated = None
        self.monster_spawn_timer = None
        self.monster_spawn_rate = None
        self.monster_base_speed = None
        self.combo_counter = None
        self.combo_timer = None
        self.screen_flash_timer = None
        
        # This is called to set up the initial state, but the user must call reset() before the first step()
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.GROUND_Y)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = True
        self.player_facing_right = True
        self.player_attack_timer = 0
        self.player_attack_cooldown = 0
        self.player_invincibility_timer = 0
        
        # Game state
        self.steps = 0
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.monsters = []
        self.particles = []
        self.camera_x = 0
        self.monsters_defeated = 0
        self.monster_spawn_timer = 60
        self.monster_spawn_rate = 120 # Frames between spawns
        self.monster_base_speed = 1.5
        self.combo_counter = 0
        self.combo_timer = 0
        self.screen_flash_timer = 0
        
        # Initialize parallax stars
        for layer in self.parallax_layers:
            layer["stars"] = []
            for _ in range(100):
                layer["stars"].append((self.np_random.integers(0, self.WIDTH*2), self.np_random.integers(0, self.GROUND_Y)))

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        if not self.game_over:
            # Idle penalty
            if movement == 0:
                reward -= 0.02

            # Horizontal Movement
            if movement == 3: # Left
                self.player_vel.x = -4
                self.player_facing_right = False
            elif movement == 4: # Right
                self.player_vel.x = 4
                self.player_facing_right = True
            
            # Jumping
            if movement == 1 and self.player_on_ground:
                self.player_vel.y = -12
                self.player_on_ground = False
                # sfx: jump

            # Attacking
            if space_held and self.player_attack_cooldown == 0:
                self.player_attack_timer = 8 # Attack lasts for 8 frames
                self.player_attack_cooldown = 20 # Can't attack for 20 frames
                # sfx: sword_swing

        # --- Game Logic Update ---
        self._update_player()
        self._update_monsters()
        self._update_particles()
        self._spawn_monsters()

        # --- Collision Detection & Rewards ---
        reward += self._handle_collisions()

        # --- Timers and Cooldowns ---
        self.player_attack_timer = max(0, self.player_attack_timer - 1)
        self.player_attack_cooldown = max(0, self.player_attack_cooldown - 1)
        self.player_invincibility_timer = max(0, self.player_invincibility_timer - 1)
        self.screen_flash_timer = max(0, self.screen_flash_timer - 1)
        self.combo_timer = max(0, self.combo_timer - 1)
        if self.combo_timer == 0:
            self.combo_counter = 0

        # --- Termination Conditions ---
        self.steps += 1
        if self.lives <= 0 or self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        if self.monsters_defeated >= self.WIN_CONDITION:
            terminated = True
            self.game_over = True
            reward += 50 # Victory reward

        truncated = False # This environment does not truncate based on time limits

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self):
        # Apply physics
        self.player_vel.y += self.GRAVITY
        self.player_pos += self.player_vel
        self.player_vel.x *= 0.8 # Friction

        # Ground collision
        if self.player_pos.y >= self.GROUND_Y:
            self.player_pos.y = self.GROUND_Y
            self.player_vel.y = 0
            if not self.player_on_ground: # Landing effect
                self._create_particles(pygame.Vector2(self.player_pos.x, self.player_pos.y + 10), 5, (200, 200, 200))
            self.player_on_ground = True
        
        # Screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 10, self.WIDTH - 10)

    def _update_monsters(self):
        for monster in self.monsters[:]:
            # Move towards player
            if (self.player_pos - monster['pos']).length() > 0:
                direction = (self.player_pos - monster['pos']).normalize()
            else:
                direction = pygame.Vector2(0)
            monster['pos'] += direction * monster['speed']
            
            # Simple jump for ground monsters
            if monster['type'] == 'ground' and monster['pos'].y >= self.GROUND_Y and self.np_random.random() < 0.01:
                monster['vel'].y = -8
            
            # Gravity for ground monsters
            if monster['type'] == 'ground':
                monster['vel'].y += self.GRAVITY
                monster['pos'] += monster['vel']
                if monster['pos'].y > self.GROUND_Y:
                    monster['pos'].y = self.GROUND_Y
                    monster['vel'].y = 0

            monster['hit_timer'] = max(0, monster['hit_timer'] - 1)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _spawn_monsters(self):
        self.monster_spawn_timer -= 1
        if self.monster_spawn_timer <= 0:
            side = self.np_random.integers(0, 2)
            x = -20 if side == 0 else self.WIDTH + 20
            
            monster_type = 'ground' if self.np_random.random() < 0.7 else 'flying'
            y = self.GROUND_Y if monster_type == 'ground' else self.np_random.integers(100, self.GROUND_Y - 50)
            
            speed_increase = (self.monsters_defeated // 5) * 0.05
            speed = self.monster_base_speed + speed_increase + self.np_random.uniform(-0.2, 0.2)
            
            self.monsters.append({
                'pos': pygame.Vector2(x, y),
                'vel': pygame.Vector2(0, 0),
                'health': 100,
                'type': monster_type,
                'color': self.COLOR_MONSTER_1 if monster_type == 'ground' else self.COLOR_MONSTER_2,
                'size': 20,
                'speed': speed,
                'hit_timer': 0,
            })
            self.monster_spawn_timer = max(30, self.monster_spawn_rate - self.monsters_defeated * 2)

    def _handle_collisions(self):
        reward = 0
        
        # Player attack vs Monsters
        if self.player_attack_timer > 0:
            attack_dir = 1 if self.player_facing_right else -1
            attack_rect = pygame.Rect(
                self.player_pos.x + (20 * attack_dir) - (25 if attack_dir == -1 else 0),
                self.player_pos.y - 30, 45, 40
            )
            for monster in self.monsters:
                monster_rect = pygame.Rect(monster['pos'].x - monster['size']/2, monster['pos'].y - monster['size'], monster['size'], monster['size'])
                if monster_rect.colliderect(attack_rect) and monster['health'] > 0:
                    monster['health'] = 0 # One-hit kills
                    monster['hit_timer'] = 10
                    reward += 0.1 # Hit reward
                    # sfx: monster_hit

        # Check for defeated monsters
        for monster in self.monsters[:]:
            if monster['health'] <= 0:
                # sfx: monster_die
                self._create_particles(monster['pos'], 20, monster['color'])
                self.monsters.remove(monster)
                self.monsters_defeated += 1
                self.score += 100
                reward += 1 # Defeat reward
                
                # Combo bonus
                reward += 0.5 * self.combo_counter
                self.score += 50 * self.combo_counter
                self.combo_counter += 1
                self.combo_timer = 120 # 2 seconds at 60fps, 4 at 30fps
        
        # Player vs Monsters
        if self.player_invincibility_timer == 0:
            player_rect = pygame.Rect(self.player_pos.x - 7, self.player_pos.y - 30, 14, 30)
            for monster in self.monsters:
                monster_rect = pygame.Rect(monster['pos'].x - monster['size']/2, monster['pos'].y - monster['size'], monster['size'], monster['size'])
                if player_rect.colliderect(monster_rect):
                    self.lives -= 1
                    reward -= 1 # Life loss penalty
                    self.player_invincibility_timer = 90 # 3 seconds of invincibility
                    self.screen_flash_timer = 10
                    self.combo_counter = 0
                    # sfx: player_hurt
                    break
        return reward

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(pos), # FIX: pygame.Vector2 objects are copied by creating a new one
                'vel': pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-5, 1)),
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
            })

    def _get_observation(self):
        # --- Camera Update ---
        target_camera_x = self.player_pos.x - self.WIDTH / 2
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_ground()
        self._render_particles()
        self._render_monsters()
        self._render_player()
        self._render_ui()

        if self.screen_flash_timer > 0:
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = 150 * (self.screen_flash_timer / 10)
            flash_surface.fill((*self.COLOR_SCREEN_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for layer in self.parallax_layers:
            for x, y in layer['stars']:
                screen_x = (x - self.camera_x * layer['speed']) % self.WIDTH
                size_x, size_y = layer['size']
                pygame.draw.rect(self.screen, layer['color'], (int(screen_x), int(y), size_x, size_y))

    def _render_ground(self):
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

    def _render_player(self):
        screen_pos_x = int(self.player_pos.x - self.camera_x)
        screen_pos_y = int(self.player_pos.y)

        # Body
        color = self.COLOR_PLAYER
        if self.player_invincibility_timer > 0 and (self.steps // 3) % 2 == 0:
            color = (0,0,0,0) # Flicker when invincible
        
        player_rect = pygame.Rect(screen_pos_x - 7, screen_pos_y - 30, 14, 30)
        pygame.draw.rect(self.screen, color, player_rect, border_radius=3)
        
        # Attack animation
        if self.player_attack_timer > 0:
            progress = self.player_attack_timer / 8.0
            angle_start = -math.pi/2 - (math.pi/2 * (1-progress))
            angle_end = math.pi/2 + (math.pi/2 * (1-progress))
            if not self.player_facing_right:
                angle_start += math.pi
                angle_end += math.pi
            
            attack_center_x = screen_pos_x + (15 if self.player_facing_right else -15)
            
            for i in range(4):
                radius = 20 + i * 2
                pygame.gfxdraw.arc(self.screen, attack_center_x, screen_pos_y - 15, radius, int(math.degrees(angle_start)), int(math.degrees(angle_end)), self.COLOR_PLAYER_ATTACK)

    def _render_monsters(self):
        for monster in self.monsters:
            screen_pos_x = int(monster['pos'].x - self.camera_x)
            screen_pos_y = int(monster['pos'].y)
            color = monster['color'] if monster['hit_timer'] == 0 else self.COLOR_HIT_FLASH
            
            size = monster['size']
            if monster['type'] == 'ground':
                pygame.draw.rect(self.screen, color, (screen_pos_x - size/2, screen_pos_y - size, size, size), border_radius=4)
            else: # flying
                pygame.gfxdraw.aacircle(self.screen, screen_pos_x, screen_pos_y-size//2, int(size//2), color)
                pygame.gfxdraw.filled_circle(self.screen, screen_pos_x, screen_pos_y-size//2, int(size//2), color)


    def _render_particles(self):
        for p in self.particles:
            screen_pos_x = int(p['pos'].x - self.camera_x)
            screen_pos_y = int(p['pos'].y)
            size = max(1, int(p['lifespan'] / 5))
            pygame.draw.rect(self.screen, p['color'], (screen_pos_x, screen_pos_y, size, size))

    def _render_ui(self):
        # Lives
        for i in range(self.lives):
            pygame.gfxdraw.filled_circle(self.screen, 30 + i * 35, 30, 10, self.COLOR_MONSTER_1)
            pygame.gfxdraw.aacircle(self.screen, 30 + i * 35, 30, 10, self.COLOR_MONSTER_1)

        # Score
        score_text = self.game_font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))

        # Combo
        if self.combo_counter > 1:
            combo_text = self.combo_font.render(f"x{self.combo_counter} Combo!", True, self.COLOR_PLAYER)
            self.screen.blit(combo_text, (self.WIDTH - combo_text.get_width() - 20, 55))

        # Monsters Defeated
        monster_text = self.combo_font.render(f"Defeated: {self.monsters_defeated}/{self.WIN_CONDITION}", True, self.COLOR_TEXT)
        self.screen.blit(monster_text, (self.WIDTH - monster_text.get_width() - 20, 85))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            status_text = "VICTORY!" if self.monsters_defeated >= self.WIN_CONDITION else "GAME OVER"
            status_render = self.game_font.render(status_text, True, self.COLOR_PLAYER if self.monsters_defeated >= self.WIN_CONDITION else self.COLOR_MONSTER_1)
            text_rect = status_render.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(status_render, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "monsters_defeated": self.monsters_defeated,
            "combo": self.combo_counter,
        }

    def close(self):
        pygame.quit()
        

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To view the game, comment out the os.environ line at the top of the file
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("GameEnv")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # Action defaults
        movement = 0 # none
        space_held = 0 # released
        shift_held = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2

        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()