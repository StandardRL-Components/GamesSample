
# Generated: 2025-08-27T16:40:34.686480
# Source Brief: brief_01296.md
# Brief Index: 1296

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A side-scrolling action game where a player controls a jumping robot, blasting enemy robots to survive and progress through increasingly difficult stages."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.TOTAL_STAGES = 3
        self.ENEMIES_PER_STAGE = 20
        self.INITIAL_LIVES = 5

        # Physics constants
        self.PLAYER_SPEED = 6
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -14
        self.PLAYER_SHOOT_COOLDOWN = 6  # frames
        self.PROJECTILE_SPEED = 12

        # Color palette
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_PLATFORM = (70, 80, 100)
        self.COLOR_PLAYER = (60, 150, 255)
        self.COLOR_PLAYER_JETPACK = (255, 200, 0)
        self.COLOR_ENEMY = (255, 80, 60)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (255, 150, 0)
        self.COLOR_TEXT = (240, 240, 240)
        
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
        self.font_main = pygame.font.SysFont("Consolas", 24)
        self.font_stage = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Platforms definition (x, y, width, height)
        self.platforms = [
            pygame.Rect(0, self.HEIGHT - 20, self.WIDTH, 20),
            pygame.Rect(100, 280, 200, 20),
            pygame.Rect(340, 280, 200, 20),
            pygame.Rect(220, 160, 200, 20),
        ]
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = None
        self.player_shoot_timer = None
        self.player_lives = None
        self.current_stage = None
        self.enemies_destroyed_in_stage = None
        self.enemies = None
        self.player_projectiles = None
        self.enemy_projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_vel = [0, 0]
        self.on_ground = False
        self.player_shoot_timer = 0
        self.player_lives = self.INITIAL_LIVES
        self.current_stage = 1
        self.enemies_destroyed_in_stage = 0
        
        self.enemies = []
        self._spawn_enemies()

        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.steps += 1

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        # --- Handle player actions ---
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] = 0
        
        # Jumping
        if movement == 1 and self.on_ground:  # Up
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: player_jump.wav
            
        # Shooting
        if space_held and self.player_shoot_timer <= 0:
            self._fire_player_projectile()
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
            # sfx: player_shoot.wav
        else:
            reward -= 0.005 # Small penalty for not shooting to encourage action

        if self.player_shoot_timer > 0:
            self.player_shoot_timer -= 1
            
        # --- Update game state ---
        self._update_player()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        # --- Handle collisions and events ---
        reward += self._handle_collisions()
        
        # --- Stage progression ---
        if self.enemies_destroyed_in_stage >= self.ENEMIES_PER_STAGE:
            self.current_stage += 1
            reward += 50
            if self.current_stage > self.TOTAL_STAGES:
                self.game_over = True # Victory
                reward += 100
            else:
                self.enemies_destroyed_in_stage = 0
                self._spawn_enemies()

        # --- Check termination conditions ---
        terminated = self.game_over or self.player_lives <= 0 or self.steps >= self.MAX_STEPS
        if self.player_lives <= 0 and not self.game_over:
             self.game_over = True # Defeat

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_enemies(self):
        self.enemies.clear()
        spawn_platforms = self.platforms[1:] # Don't spawn on the floor
        for _ in range(self.ENEMIES_PER_STAGE):
            platform = random.choice(spawn_platforms)
            x = random.randint(platform.left, platform.right - 20)
            y = platform.top - 20
            fire_rate = 0.5 + (self.current_stage - 1) * 0.05
            shoot_cooldown = int(self.FPS / fire_rate)
            self.enemies.append({
                "pos": [x, y],
                "size": [20, 20],
                "shoot_timer": random.randint(0, shoot_cooldown),
                "shoot_cooldown": shoot_cooldown,
                "pulse_phase": random.uniform(0, 2 * math.pi)
            })

    def _update_player(self):
        # Apply physics
        self.player_pos[0] += self.player_vel[0]
        self.player_vel[1] += self.GRAVITY
        self.player_pos[1] += self.player_vel[1]
        
        # Screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH - 20)
        
        # Platform collision
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 20, 30)
        for plat in self.platforms:
            if player_rect.colliderect(plat) and self.player_vel[1] > 0:
                # Check if player was above the platform in the previous frame
                if player_rect.bottom - self.player_vel[1] <= plat.top:
                    self.player_pos[1] = plat.top - 30
                    self.player_vel[1] = 0
                    self.on_ground = True
                    break

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy["shoot_timer"] -= 1
            if enemy["shoot_timer"] <= 0:
                self._fire_enemy_projectile(enemy)
                enemy["shoot_timer"] = enemy["shoot_cooldown"]
                # sfx: enemy_shoot.wav
            enemy["pulse_phase"] += 0.1

    def _update_projectiles(self):
        # Player projectiles
        for p in self.player_projectiles[:]:
            p['pos'][1] -= self.PROJECTILE_SPEED
            if p['pos'][1] < 0:
                self.player_projectiles.remove(p)
        # Enemy projectiles
        for p in self.enemy_projectiles[:]:
            p['pos'][1] += self.PROJECTILE_SPEED
            if p['pos'][1] > self.HEIGHT:
                self.enemy_projectiles.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                
    def _fire_player_projectile(self):
        self.player_projectiles.append({
            "pos": [self.player_pos[0] + 8, self.player_pos[1]]
        })

    def _fire_enemy_projectile(self, enemy):
        self.enemy_projectiles.append({
            "pos": [enemy['pos'][0] + enemy['size'][0] / 2 - 2, enemy['pos'][1] + enemy['size'][1]]
        })

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 20, 30)
        
        # Player projectiles vs. enemies
        for p_proj in self.player_projectiles[:]:
            p_rect = pygame.Rect(p_proj['pos'][0], p_proj['pos'][1], 4, 10)
            for enemy in self.enemies[:]:
                e_rect = pygame.Rect(enemy['pos'][0], enemy['pos'][1], enemy['size'][0], enemy['size'][1])
                if p_rect.colliderect(e_rect):
                    self._create_explosion(enemy['pos'])
                    self.enemies.remove(enemy)
                    self.player_projectiles.remove(p_proj)
                    self.score += 10
                    self.enemies_destroyed_in_stage += 1
                    reward += 1
                    # sfx: enemy_explosion.wav
                    break
        
        # Enemy projectiles vs. player
        for e_proj in self.enemy_projectiles[:]:
            e_rect = pygame.Rect(e_proj['pos'][0], e_proj['pos'][1], 4, 10)
            if player_rect.colliderect(e_rect):
                self.enemy_projectiles.remove(e_proj)
                self.player_lives -= 1
                self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                # sfx: player_hit.wav
                break # only take one hit per frame
                
        return reward

    def _create_explosion(self, pos, color=(255, 120, 0), count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(10, 20),
                'color': color
            })
            
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Platforms
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            
        # Particles
        for p in self.particles:
            size = max(0, int(p['life'] / 4))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

        # Enemy projectiles
        for p in self.enemy_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJ, (p['pos'][0], p['pos'][1], 4, 10))
            
        # Player projectiles
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (p['pos'][0], p['pos'][1], 4, 10))
            
        # Enemies
        for enemy in self.enemies:
            pulse = (math.sin(enemy['pulse_phase']) + 1) / 2 * 5
            color = tuple(np.clip([c + pulse * 2 for c in self.COLOR_ENEMY], 0, 255))
            e_rect = pygame.Rect(enemy['pos'][0], enemy['pos'][1], enemy['size'][0], enemy['size'][1])
            pygame.draw.rect(self.screen, color, e_rect)
            
        # Player
        if self.player_lives > 0:
            player_rect = pygame.Rect(int(self.player_pos[0]), int(self.player_pos[1]), 20, 30)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
            # Jetpack effect when jumping
            if self.player_vel[1] < 0:
                for i in range(5):
                    jet_x = player_rect.centerx + random.uniform(-4, 4)
                    jet_y = player_rect.bottom + random.uniform(0, 10)
                    jet_size = random.randint(1, 4)
                    pygame.draw.circle(self.screen, self.COLOR_PLAYER_JETPACK, (int(jet_x), int(jet_y)), jet_size)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_main.render(f"LIVES: {self.player_lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))
        
        # Stage
        if not self.game_over:
            stage_text = self.font_stage.render(f"STAGE {self.current_stage}", True, self.COLOR_TEXT)
            self.screen.blit(stage_text, (self.WIDTH // 2 - stage_text.get_width() // 2, self.HEIGHT - 40))
        elif self.current_stage > self.TOTAL_STAGES:
            end_text = self.font_stage.render("VICTORY!", True, self.COLOR_PLAYER_PROJ)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - 16))
        else:
            end_text = self.font_stage.render("GAME OVER", True, self.COLOR_ENEMY)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - 16))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.current_stage
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
    env = GameEnv(render_mode="rgb_array")
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Blaster")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        # Action mapping for human keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2 # not used
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Event handling for closing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        clock.tick(env.FPS)
        
    env.close()
    print(f"Game Over! Final Info: {info}")