
# Generated: 2025-08-27T18:15:11.737029
# Source Brief: brief_01772.md
# Brief Index: 1772

        
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


# Helper classes for game objects
class Player:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.size = pygame.Vector2(20, 30)
        self.on_ground = False
        self.jump_squash = 0  # For animation

    def get_rect(self):
        # Adjust size for squash/stretch animation
        squash_factor = self.jump_squash * 0.2
        width = self.size.x * (1 - squash_factor)
        height = self.size.y * (1 + squash_factor)
        return pygame.Rect(self.pos.x, self.pos.y, width, height)

class Platform:
    def __init__(self, x, y, width, height, move_range=0, move_speed=0, move_offset=0):
        self.base_pos = pygame.Vector2(x, y)
        self.pos = pygame.Vector2(x, y)
        self.size = pygame.Vector2(width, height)
        self.move_range = move_range
        self.move_speed = move_speed
        self.move_offset = move_offset

    def update(self, steps):
        if self.move_range > 0:
            self.pos.y = self.base_pos.y + math.sin(steps * self.move_speed + self.move_offset) * self.move_range

    def get_rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.size.x, self.size.y)

class Enemy:
    def __init__(self, x, y, patrol_range, speed):
        self.pos = pygame.Vector2(x, y)
        self.size = pygame.Vector2(25, 25)
        self.start_x = x
        self.patrol_range = patrol_range
        self.speed = speed
        self.direction = 1

    def update(self):
        self.pos.x += self.speed * self.direction
        if self.pos.x > self.start_x + self.patrol_range or self.pos.x < self.start_x:
            self.direction *= -1
            self.pos.x += self.speed * self.direction # Prevent getting stuck

    def get_rect(self):
        return pygame.Rect(self.pos.x, self.pos.y, self.size.x, self.size.y)

class Particle:
    def __init__(self, x, y, vx, vy, life, color, size):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(vx, vy)
        self.life = life
        self.color = color
        self.size = size

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.size = max(0, self.size - 0.1)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: ←→ to move, ↑ to jump. Reach the yellow flag to win!"
    game_description = "A fast-paced pixel-art platformer. Evade red enemies and navigate moving platforms to reach the goal before time runs out."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    LEVEL_WIDTH = 1920  # 3 screens wide
    FPS = 60
    GRAVITY = 0.4
    JUMP_STRENGTH = -9
    PLAYER_SPEED = 4

    # --- Colors ---
    COLOR_BG = (135, 206, 235) # Light Sky Blue
    COLOR_PLAYER = (57, 255, 20) # Neon Green
    COLOR_ENEMY = (255, 69, 0) # Red-Orange
    COLOR_PLATFORM = (105, 105, 105) # Dim Gray
    COLOR_GOAL = (255, 215, 0) # Gold
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)

        self.level_definitions = [
            # Level 1
            {
                "platforms": [
                    (0, 350, 300, 50),
                    (400, 320, 150, 30, 20, 0.03),
                    (650, 280, 100, 20),
                    (850, 320, 150, 30, 20, 0.03, math.pi),
                    (1100, 250, 200, 30),
                    (1400, 200, 100, 20, 40, 0.05),
                    (1600, 350, 300, 50),
                ],
                "enemies": [(1150, 225, 50)],
                "goal_pos": (1750, 300)
            },
            # Level 2
            {
                "platforms": [
                    (0, 350, 200, 50),
                    (300, 300, 100, 20, 50, 0.04),
                    (500, 250, 100, 20),
                    (700, 300, 100, 20, 50, 0.04, math.pi),
                    (900, 200, 80, 20),
                    (1100, 150, 80, 20, 60, 0.06),
                    (1300, 200, 80, 20),
                    (1500, 250, 80, 20, 60, 0.06, math.pi),
                    (1700, 350, 220, 50),
                ],
                "enemies": [(500, 225, 0), (1750, 325, 50)],
                "goal_pos": (1800, 300)
            },
            # Level 3
            {
                "platforms": [
                    (0, 350, 150, 50),
                    (250, 300, 80, 20, 60, 0.08),
                    (450, 220, 80, 20),
                    (650, 150, 80, 20, 60, 0.08, math.pi),
                    (850, 220, 80, 20),
                    (1050, 300, 80, 20, 60, 0.08),
                    (1250, 250, 50, 20),
                    (1450, 180, 50, 20),
                    (1650, 350, 270, 50)
                ],
                "enemies": [(900, 325, 200), (1350, 325, 200)],
                "goal_pos": (1800, 300)
            }
        ]
        
        self.reset()

    def _setup_level(self):
        level_data = self.level_definitions[self.current_level]
        
        self.platforms = [Platform(*p) for p in level_data["platforms"]]
        
        current_enemy_speed = self.base_enemy_speed + 0.5 * self.current_level
        self.enemies = [Enemy(e[0], e[1], e[2], current_enemy_speed) for e in level_data["enemies"]]
        
        goal_x, goal_y = level_data["goal_pos"]
        self.goal_rect = pygame.Rect(goal_x, goal_y, 20, 50)
        
        self.player = Player(100, 250)
        self.last_platform_idx = -1
        self.time_on_platform = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminated = False
        self.total_time_steps = self.FPS * 180 # 3 minutes total
        
        self.current_level = 0
        self.base_enemy_speed = 1.0
        
        self._setup_level()
        
        self.camera_x = 0
        self.particles = []
        
        # Parallax background clouds
        self.clouds = []
        for _ in range(20):
            self.clouds.append({
                "pos": pygame.Vector2(self.np_random.uniform(0, self.LEVEL_WIDTH), self.np_random.uniform(20, 150)),
                "size": self.np_random.uniform(50, 150),
                "depth": self.np_random.uniform(0.1, 0.6) # Slower-moving clouds are further away
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.terminated, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        reward = 0.01  # Survival reward

        # 1. Process Input and Update Player
        self._handle_input(action)
        self._update_player()
        
        # 2. Update Game World
        for p in self.platforms:
            p.update(self.steps)
        for e in self.enemies:
            e.update()
        self._update_particles()
        
        # 3. Handle Collisions and Game Logic
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # 4. Check Termination Conditions
        self.terminated, termination_reward = self._check_termination()
        reward += termination_reward
        if self.terminated:
            self.game_over = True
            
        # 5. Update Camera
        self._update_camera()

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            self.terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement, _, _ = action
        
        # Horizontal movement
        if movement == 3: # Left
            self.player.vel.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player.vel.x = self.PLAYER_SPEED
        else:
            self.player.vel.x = 0
            
        # Jump
        if movement == 1 and self.player.on_ground:
            self.player.vel.y = self.JUMP_STRENGTH
            self.player.on_ground = False
            self.player.jump_squash = 1.0 # Start jump animation
            # sfx: jump
            self._spawn_particles(self.player.get_rect().midbottom, 5)

    def _update_player(self):
        # Apply gravity
        self.player.vel.y += self.GRAVITY
        self.player.vel.y = min(self.player.vel.y, 10) # Terminal velocity

        # Move player
        self.player.pos += self.player.vel

        # Clamp player to level bounds
        self.player.pos.x = max(0, min(self.player.pos.x, self.LEVEL_WIDTH - self.player.size.x))

        # Update jump animation
        if self.player.jump_squash > 0:
            self.player.jump_squash -= 0.1
        else:
            self.player.jump_squash = 0

    def _handle_collisions(self):
        reward = 0
        player_rect = self.player.get_rect()
        was_on_ground = self.player.on_ground
        self.player.on_ground = False
        
        # Player vs Platforms
        current_platform_idx = -1
        for i, p in enumerate(self.platforms):
            platform_rect = p.get_rect()
            if player_rect.colliderect(platform_rect):
                # Check if player is landing on top
                if self.player.vel.y > 0 and player_rect.bottom < platform_rect.top + self.player.vel.y + 1:
                    self.player.pos.y = platform_rect.top - player_rect.height
                    self.player.vel.y = 0
                    self.player.on_ground = True
                    current_platform_idx = i
                    if not was_on_ground: # Just landed
                        self.player.jump_squash = -0.8 # Land squash animation
                        # sfx: land
                    break
        
        # Platform-related rewards
        if self.player.on_ground:
            if current_platform_idx != self.last_platform_idx:
                reward += 5  # Reached a new platform
                self.time_on_platform = 0
                self.last_platform_idx = current_platform_idx
            else:
                self.time_on_platform += 1
                if self.time_on_platform > self.FPS * 2: # > 2 seconds
                    reward -= 0.2
                    self.time_on_platform = 0 # Reset to avoid constant penalty
        else:
            self.last_platform_idx = -1
            self.time_on_platform = 0

        # Player vs Enemies
        for e in self.enemies:
            if player_rect.colliderect(e.get_rect()):
                self.terminated = True
                # sfx: player_hit
                return -100 # Enemy collision penalty

        # Player vs Goal
        if player_rect.colliderect(self.goal_rect):
            self.current_level += 1
            # sfx: level_complete
            if self.current_level >= len(self.level_definitions):
                self.terminated = True # Won the game
                return 100 # Victory reward
            else:
                self._setup_level() # Go to next level
                # No terminal reward here, just progress
        
        # Player vs World bottom
        if player_rect.top > self.SCREEN_HEIGHT:
            self.terminated = True
            # sfx: fall
            return -5 # Fall penalty (will be combined with timeout)
            
        return reward

    def _check_termination(self):
        if self.terminated: # Already terminated by collision
            return True, 0

        if self.steps >= self.total_time_steps:
            # sfx: timeout
            return True, -50 # Timeout penalty
        
        return False, 0
        
    def _update_camera(self):
        # Center camera on player, with clamping
        target_camera_x = self.player.pos.x - self.SCREEN_WIDTH / 2
        self.camera_x = max(0, min(target_camera_x, self.LEVEL_WIDTH - self.SCREEN_WIDTH))
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        # Render game elements relative to camera
        self._render_platforms()
        self._render_enemies()
        self._render_goal()
        self._render_player()
        self._render_particles()

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for cloud in self.clouds:
            # Use modulo for infinite wrapping
            x = (cloud["pos"].x - self.camera_x * cloud["depth"]) % self.LEVEL_WIDTH
            # If it wraps, draw it twice for seamless transition
            if x > self.SCREEN_WIDTH - cloud["size"]:
                 pygame.gfxdraw.filled_circle(self.screen, int(x - self.LEVEL_WIDTH), int(cloud["pos"].y), int(cloud["size"]/2), (255,255,255, 80))
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(cloud["pos"].y), int(cloud["size"]/2), (255,255,255, 80))

    def _render_platforms(self):
        for p in self.platforms:
            rect = p.get_rect()
            rect.x -= self.camera_x
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, rect)

    def _render_enemies(self):
        for e in self.enemies:
            rect = e.get_rect()
            rect.x -= self.camera_x
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect)
            pygame.draw.rect(self.screen, (0,0,0), rect, 2) # Outline

    def _render_goal(self):
        goal_world_rect = self.goal_rect.copy()
        goal_screen_rect = goal_world_rect.move(-self.camera_x, 0)
        # Pole
        pygame.draw.line(self.screen, (150, 150, 150), goal_screen_rect.bottomleft, (goal_screen_rect.x, goal_screen_rect.y + 100), 5)
        # Flag
        pygame.draw.polygon(self.screen, self.COLOR_GOAL, [goal_screen_rect.topleft, (goal_screen_rect.right + 20, goal_screen_rect.centery), goal_screen_rect.bottomleft])

    def _render_player(self):
        player_rect = self.player.get_rect()
        player_rect.x -= self.camera_x
        # Glow effect
        glow_rect = player_rect.inflate(10, 10)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surface, (255, 255, 255, 60), glow_surface.get_rect())
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        pygame.draw.rect(self.screen, (0,0,0), player_rect, 2, border_radius=3) # Outline

    def _spawn_particles(self, pos, count):
        for _ in range(count):
            vx = self.np_random.uniform(-1, 1)
            vy = self.np_random.uniform(0.5, 1.5)
            life = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append(Particle(pos[0], pos[1], vx, vy, life, (200,200,200), size))

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]

    def _render_particles(self):
        for p in self.particles:
            pos_x = p.pos.x - self.camera_x
            pygame.draw.circle(self.screen, p.color, (int(pos_x), int(p.pos.y)), int(p.size))

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.total_time_steps - self.steps) // self.FPS)
        time_text = f"Time: {time_left}"
        self._draw_text(time_text, (self.SCREEN_WIDTH - 100, 20), self.font_ui)
        
        # Score
        score_text = f"Score: {int(self.score)}"
        self._draw_text(score_text, (80, 20), self.font_ui)

        # Level
        level_text = f"Level: {self.current_level + 1} / {len(self.level_definitions)}"
        self._draw_text(level_text, (self.SCREEN_WIDTH / 2, 20), self.font_ui)

        if self.game_over:
            if self.current_level >= len(self.level_definitions):
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2), self.font_game_over)

    def _draw_text(self, text, center_pos, font):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect(center=center_pos)
        shadow_rect = shadow_surf.get_rect(center=(center_pos[0]+2, center_pos[1]+2))
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level + 1
        }
        
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    # Manual play loop
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for display ---
    pygame.display.set_caption("Platformer Game")
    screen_display = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop
    running = True
    total_reward = 0
    
    # Map keys to actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
    }

    while running:
        # Default action is NO-OP
        action = [0, 0, 0] # movement=none, space=released, shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get pressed keys for continuous actions
        keys = pygame.key.get_pressed()
        
        # Movement (prioritize one direction if both pressed)
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        if keys[pygame.K_UP]:
            action[0] = 1 # Jump is separate from horizontal
            
        # Space and Shift
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Info: {info}")
            total_reward = 0
            obs, info = env.reset()

    env.close()