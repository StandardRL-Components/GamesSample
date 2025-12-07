import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Zombie:
    def __init__(self, x, y, speed, id):
        self.id = id
        self.width = 20
        self.height = 40
        self.rect = pygame.Rect(x, y - self.height, self.width, self.height)
        self.speed = speed
        self.animation_state = 0

class Obstacle:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y - height, width, height)

class Particle:
    def __init__(self, x, y, color, life):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-0.5, -1.5)
        self.life = life
        self.max_life = life
        self.color = color
        self.radius = random.randint(2, 4)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Press Space to jump and avoid the zombies."
    )

    game_description = (
        "A fast-paced side-scrolling survival game. Run and jump to survive the zombie horde for as long as you can across three increasingly difficult stages."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GROUND_Y = 350
        self.WORLD_CHUNK_SIZE = 2000

        # Colors
        self.COLOR_BG = (15, 25, 40)
        self.COLOR_GROUND = (87, 65, 50)
        self.COLOR_PLAYER = (50, 205, 50)
        self.COLOR_PLAYER_GLOW = (152, 251, 152, 50)
        self.COLOR_ZOMBIE = (220, 20, 60)
        self.COLOR_OBSTACLE = (100, 100, 110)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_STAR = (200, 200, 220)

        # Physics constants
        self.GRAVITY = 0.6
        self.PLAYER_SPEED = 4.0
        self.JUMP_STRENGTH = -12

        # Game parameters
        self.FPS = 30
        self.MAX_STAGES = 3
        self.STAGE_DURATION_SECS = 60

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_on_ground = None
        self.player_rect = None
        
        self.zombies = None
        self.obstacles = None
        self.particles = None
        self.stars = None
        
        self.camera_x = None
        self.steps = None
        self.score = None
        self.survival_time = None
        self.stage = None
        self.game_over = None
        
        self.zombie_spawn_rate = None
        self.zombie_speed = None
        self.zombie_id_counter = None
        self.passed_zombies = None
        self.last_obstacle_x = None
        
        # Generate persistent background stars
        self.stars = [
            (random.randint(0, self.SCREEN_WIDTH * 2), random.randint(0, self.GROUND_Y - 50), random.randint(1, 2))
            for _ in range(150)
        ]
        
        # This is called here to set up np_random, but the actual reset logic is below.
        # The full state is reset in the reset() method itself.
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(100, self.GROUND_Y - 40)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = True
        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 30, 40)
        
        # Game objects
        self.zombies = []
        self.obstacles = []
        self.particles = []
        
        # Game flow state
        self.camera_x = 0
        self.steps = 0
        self.score = 0
        self.survival_time = 0.0
        self.stage = 1
        self.game_over = False
        
        # Difficulty state
        self.zombie_spawn_rate = 1 / (self.FPS * 2) # 1 every 2 seconds
        self.zombie_speed = 1.0
        self.zombie_id_counter = 0
        self.passed_zombies = set()
        
        # Procedural generation
        # Create a safe starting area to pass stability tests. The player moves at 4px/step,
        # so in 60 steps, they travel 240px. Starting obstacles at x=400 ensures no
        # collision within this period. (chunk_start + 200 = 400 -> chunk_start=200)
        self.last_obstacle_x = 200
        self._generate_world_chunk()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Handle input
            space_pressed = action[1] == 1
            if space_pressed and self.player_on_ground:
                self.player_vel.y = self.JUMP_STRENGTH
                self.player_on_ground = False
                # Sound effect placeholder: # pygame.mixer.Sound.play(jump_sound)

            # Update game logic
            self._update_player()
            reward_from_zombies, zombie_collision = self._update_zombies()
            obstacle_collision = self._check_obstacle_collisions()
            
            self._update_particles()
            self._update_world()

            # Calculate rewards and check termination
            reward += 0.1  # Survival reward
            reward += reward_from_zombies
            
            if obstacle_collision:
                reward -= 5
                self.game_over = True
                
            if zombie_collision:
                self.game_over = True

            # Stage progression
            time_in_stage = (self.steps / self.FPS)
            current_stage_time = time_in_stage - (self.stage - 1) * self.STAGE_DURATION_SECS
            
            if current_stage_time >= self.STAGE_DURATION_SECS:
                self.stage += 1
                reward += 100
                if self.stage > self.MAX_STAGES:
                    self.game_over = True # Game won
                else:
                    # Reset difficulty for new stage
                    self.zombie_spawn_rate = 1 / (self.FPS * 2)
                    self.zombie_speed = 1.0
                    # Sound effect placeholder: # pygame.mixer.Sound.play(stage_complete_sound)

        terminated = self.game_over
        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self):
        # Y-axis movement (Gravity and Jump)
        self.player_vel.y += self.GRAVITY
        self.player_pos.y += self.player_vel.y
        self.player_on_ground = False

        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 30, 40)

        # Ground collision
        if self.player_rect.bottom > self.GROUND_Y:
            self.player_rect.bottom = self.GROUND_Y
            self.player_pos.y = self.player_rect.y
            self.player_vel.y = 0
            self.player_on_ground = True

        # Obstacle top collision
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs.rect) and self.player_vel.y > 0:
                # Check if the player was above the obstacle in the previous frame
                if self.player_pos.y + self.player_rect.height - self.player_vel.y <= obs.rect.top:
                    self.player_rect.bottom = obs.rect.top
                    self.player_pos.y = self.player_rect.y
                    self.player_vel.y = 0
                    self.player_on_ground = True
                    break
        
        self.player_pos.x += self.PLAYER_SPEED
        self.player_rect.x = self.player_pos.x
        
        if self.player_on_ground and self.steps % 4 == 0:
            self.particles.append(Particle(self.player_rect.left, self.player_rect.bottom, self.COLOR_GROUND, 20))


    def _check_obstacle_collisions(self):
        for obs in self.obstacles:
            if self.player_rect.colliderect(obs.rect):
                # A true collision (game over) is when running into the side, not landing on top.
                # Landing is handled in _update_player. We check if the player's bottom is
                # clearly below the obstacle's top, indicating a side impact.
                # A small tolerance (e.g., 5px) helps avoid false positives when the player
                # is perfectly aligned on top after the physics update.
                if self.player_rect.bottom > obs.rect.top + 5:
                    # Sound effect placeholder: # pygame.mixer.Sound.play(hit_sound)
                    return True
        return False

    def _update_zombies(self):
        reward = 0
        collision = False
        
        zombies_to_keep = []
        for z in self.zombies:
            z.rect.x -= z.speed
            if z.rect.right < self.camera_x: # Zombie is off-screen left
                if z.id not in self.passed_zombies:
                    reward += 1 # Reward for avoiding a zombie
                    self.passed_zombies.add(z.id)
            else:
                zombies_to_keep.append(z)
                if self.player_rect.colliderect(z.rect):
                    collision = True
                    # Sound effect placeholder: # pygame.mixer.Sound.play(player_die_sound)
                if self.steps % 15 == 0:
                    z.animation_state = 1 - z.animation_state
        self.zombies = zombies_to_keep

        if self.np_random.random() < self.zombie_spawn_rate:
            spawn_x = self.camera_x + self.SCREEN_WIDTH + 50
            self.zombie_id_counter += 1
            new_zombie = Zombie(spawn_x, self.GROUND_Y, self.zombie_speed, self.zombie_id_counter)
            
            can_spawn = True
            for obs in self.obstacles:
                if obs.rect.colliderect(new_zombie.rect):
                    can_spawn = False
                    break
            if can_spawn:
                self.zombies.append(new_zombie)
                
        return reward, collision

    def _update_world(self):
        self.camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3

        time_in_stage_secs = ((self.steps / self.FPS) % self.STAGE_DURATION_SECS)
        self.zombie_spawn_rate = min(0.1, (1 / (self.FPS * 2)) + (time_in_stage_secs * 0.005 / self.FPS))
        speed_increases = math.floor(time_in_stage_secs / 20)
        self.zombie_speed = min(2.0, 1.0 + (speed_increases * 0.01 * self.FPS))
        
        if self.player_pos.x > self.last_obstacle_x - self.SCREEN_WIDTH:
            self._generate_world_chunk()
            
    def _generate_world_chunk(self):
        chunk_start = self.last_obstacle_x
        chunk_end = chunk_start + self.WORLD_CHUNK_SIZE
        
        current_x = chunk_start + 200
        while current_x < chunk_end:
            if self.np_random.random() < 0.3:
                width = self.np_random.integers(40, 150)
                height = self.np_random.integers(30, 80)
                self.obstacles.append(Obstacle(current_x, self.GROUND_Y, width, height))
                current_x += width + self.np_random.integers(100, 250)
            else:
                current_x += self.np_random.integers(200, 400)
        
        self.last_obstacle_x = chunk_end
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x, y, size in self.stars:
            screen_x = (x - self.camera_x * 0.1) % self.SCREEN_WIDTH
            pygame.draw.rect(self.screen, self.COLOR_STAR, (screen_x, y, size, size))
            
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        for obs in self.obstacles:
            screen_rect = obs.rect.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect)

        for z in self.zombies:
            screen_rect = z.rect.move(-self.camera_x, 0)
            if screen_rect.colliderect(self.screen.get_rect()):
                pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, screen_rect)
                leg_y, leg_x1, leg_x2, leg_len = screen_rect.bottom, screen_rect.left + 5, screen_rect.right - 5, 8
                if z.animation_state == 0:
                    pygame.draw.line(self.screen, self.COLOR_ZOMBIE, (leg_x1, leg_y), (leg_x1 - 4, leg_y + leg_len), 3)
                    pygame.draw.line(self.screen, self.COLOR_ZOMBIE, (leg_x2, leg_y), (leg_x2 + 4, leg_y + leg_len), 3)
                else:
                    pygame.draw.line(self.screen, self.COLOR_ZOMBIE, (leg_x1, leg_y), (leg_x1 + 4, leg_y + leg_len), 3)
                    pygame.draw.line(self.screen, self.COLOR_ZOMBIE, (leg_x2, leg_y), (leg_x2 - 4, leg_y + leg_len), 3)

        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            screen_pos = (int(p.x - self.camera_x), int(p.y))
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], p.radius, color)

        screen_rect = self.player_rect.move(-self.camera_x, 0)
        pygame.gfxdraw.filled_circle(self.screen, screen_rect.centerx, screen_rect.centery, 30, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, screen_rect)
        pygame.draw.rect(self.screen, (255,255,255), (screen_rect.right - 8, screen_rect.top + 8, 4, 4))
        
        if self.player_on_ground:
            leg_swing = 8 * math.sin(self.steps * 0.5)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, screen_rect.midbottom, (screen_rect.centerx - leg_swing, screen_rect.bottom + 10), 4)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, screen_rect.midbottom, (screen_rect.centerx + leg_swing, screen_rect.bottom + 10), 4)
        else:
            pygame.draw.line(self.screen, self.COLOR_PLAYER, screen_rect.midbottom, (screen_rect.centerx - 4, screen_rect.bottom + 8), 4)
            pygame.draw.line(self.screen, self.COLOR_PLAYER, screen_rect.midbottom, (screen_rect.centerx + 4, screen_rect.bottom + 8), 4)

    def _render_ui(self):
        time_in_stage = ((self.steps / self.FPS) - (self.stage - 1) * self.STAGE_DURATION_SECS)
        self.survival_time = min(time_in_stage, self.STAGE_DURATION_SECS)

        stage_text = self.font_small.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        time_text = self.font_small.render(f"Time: {self.survival_time:.1f}s", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        score_text = self.font_small.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "GAME OVER"
            if self.stage > self.MAX_STAGES:
                msg = "YOU SURVIVED!"
            
            game_over_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(game_over_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "survival_time": self.survival_time
        }

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Runner")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        action = np.array([0, 0, 0])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if terminated:
            # Wait for 'r' key press to reset, handled in event loop
            pass
        else:
            obs, reward, terminated, truncated, info = env.step(action)
            
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    pygame.quit()