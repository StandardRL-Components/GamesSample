import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:08:24.095872
# Source Brief: brief_00292.md
# Brief Index: 292
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where a player navigates a neon maze.
    The goal is to collect speed-boosting orbs and reach the exit
    before the timer runs out, while avoiding walls and moving obstacles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a neon maze, collect speed-boosting orbs, and reach the exit before the timer runs out while avoiding obstacles."
    )
    user_guide = "Use the arrow keys (↑↓←→) to move your character."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    LOGIC_FPS = 30  # Assumed FPS for physics calculations

    # Colors (Neon Aesthetic)
    COLOR_BG = (10, 0, 30)
    COLOR_WALL = (50, 50, 200)
    COLOR_WALL_GLOW = (80, 80, 255)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_ORB = (255, 255, 0)
    COLOR_ORB_GLOW = (200, 200, 50)
    COLOR_EXIT = (0, 255, 100)
    COLOR_EXIT_GLOW = (100, 255, 150)
    COLOR_OBSTACLE = (255, 20, 20)
    COLOR_OBSTACLE_GLOW = (255, 100, 100)
    COLOR_UI_TEXT = (255, 255, 255)

    # Game Parameters
    MAX_TIME_SECONDS = 60
    MAX_LEVELS = 10
    PLAYER_RADIUS = 10
    PLAYER_ACCELERATION = 0.4
    PLAYER_FRICTION = 0.96
    ORB_RADIUS = 6
    PARTICLE_LIFETIME = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 28)
        except pygame.error:
            # Fallback if default font is not found
            self.font_large = pygame.font.SysFont("sans", 36)
            self.font_small = pygame.font.SysFont("sans", 28)

        # Game state variables that persist across episodes
        self.best_level = 0
        
        # Initialize state variables (will be properly set in reset)
        self.level = 0
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.speed_multiplier = 1.0
        self.walls = []
        self.orbs = []
        self.obstacles = []
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.particles = []

        # self.reset() is called by the wrapper/runner, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if 'level' in (options or {}):
            self.level = options['level']
        else:
            self.level = 1

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME_SECONDS
        self.speed_multiplier = 1.0
        
        self.player_vel = pygame.Vector2(0, 0)
        self.particles = []

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If game is over, do nothing but return final state
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        self._handle_input(movement)
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        
        # --- Handle Interactions and State Changes ---
        reward = -0.01  # Small penalty for each step to encourage speed
        self.timer -= 1.0 / self.LOGIC_FPS
        
        # Orb collection
        collected_orbs = []
        for orb in self.orbs:
            if self.player_pos.distance_to(orb['pos']) < self.PLAYER_RADIUS + orb['radius']:
                collected_orbs.append(orb)
                self.speed_multiplier *= 1.05
                reward += 0.1
                self.score += 10
                self._create_particles(orb['pos'], self.COLOR_ORB, 15)
                # Sound effect placeholder: pygame.mixer.Sound('orb_collect.wav').play()
        self.orbs = [orb for orb in self.orbs if orb not in collected_orbs]

        # --- Termination Checks ---
        terminated = False
        
        # Collision with walls
        if self._check_wall_collision():
            terminated = True
            reward = -100
            self.score -= 500
            self._create_particles(self.player_pos, self.COLOR_PLAYER, 30)
            # Sound effect placeholder: pygame.mixer.Sound('collision.wav').play()

        # Collision with obstacles
        if not terminated and self._check_obstacle_collision():
            terminated = True
            reward = -100
            self.score -= 500
            self._create_particles(self.player_pos, self.COLOR_OBSTACLE, 30)
            # Sound effect placeholder: pygame.mixer.Sound('collision.wav').play()

        # Reached exit
        if not terminated and self.exit_rect.collidepoint(self.player_pos):
            self.score += 100 * self.level
            self.best_level = max(self.best_level, self.level)
            
            if self.level >= self.MAX_LEVELS:
                reward = 100  # Big reward for winning the game
                terminated = True
            else:
                reward = 10  # Reward for completing a level
                self.level += 1
                self._generate_level() # Go to next level
                # Sound effect placeholder: pygame.mixer.Sound('level_complete.wav').play()

        # Timeout
        if self.timer <= 0:
            terminated = True
            reward = -100
            self.timer = 0
            
        self.game_over = terminated
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1:  # Up
            self.player_vel.y -= self.PLAYER_ACCELERATION
        elif movement == 2:  # Down
            self.player_vel.y += self.PLAYER_ACCELERATION
        elif movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCELERATION
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCELERATION
    
    def _update_player(self):
        self.player_vel *= self.PLAYER_FRICTION
        # Clamp velocity to a reasonable max
        if self.player_vel.length() > 15:
            self.player_vel.scale_to_length(15)
        
        self.player_pos += self.player_vel * self.speed_multiplier
        
        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _update_obstacles(self):
        for obstacle in self.obstacles:
            oscillation = obstacle['amplitude'] * math.sin(self.steps * obstacle['frequency'])
            if obstacle['axis'] == 'x':
                obstacle['rect'].centerx = obstacle['base_pos'].x + oscillation
            else:
                obstacle['rect'].centery = obstacle['base_pos'].y + oscillation

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _check_wall_collision(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_RADIUS, self.player_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        for wall in self.walls:
            if player_rect.colliderect(wall):
                # More accurate circle-rect collision
                clamped_x = max(wall.left, min(self.player_pos.x, wall.right))
                clamped_y = max(wall.top, min(self.player_pos.y, wall.bottom))
                distance = self.player_pos.distance_to((clamped_x, clamped_y))
                if distance < self.PLAYER_RADIUS:
                    return True
        return False

    def _check_obstacle_collision(self):
        for obstacle in self.obstacles:
            clamped_x = max(obstacle['rect'].left, min(self.player_pos.x, obstacle['rect'].right))
            clamped_y = max(obstacle['rect'].top, min(self.player_pos.y, obstacle['rect'].bottom))
            distance = self.player_pos.distance_to((clamped_x, clamped_y))
            if distance < self.PLAYER_RADIUS:
                return True
        return False

    def _generate_level(self):
        # Reset level-specific state
        self.walls.clear()
        self.orbs.clear()
        self.obstacles.clear()
        self.player_pos = pygame.Vector2(50, self.SCREEN_HEIGHT - 50)
        self.exit_rect = pygame.Rect(self.SCREEN_WIDTH - 60, 20, 40, 40)
        self.speed_multiplier = 1.0
        self.player_vel.update(0,0)

        # Maze generation parameters
        num_walls = 5 + self.level * 2
        num_orbs = 3 + self.level
        num_obstacles = self.level - 1
        obstacle_speed = 0.05 * self.level

        # --- Generate Layout ---
        for _ in range(num_walls):
            # Avoid placing walls on start/end points
            while True:
                w = self.np_random.integers(50, 200)
                h = self.np_random.integers(50, 200)
                x = self.np_random.integers(0, self.SCREEN_WIDTH - w)
                y = self.np_random.integers(0, self.SCREEN_HEIGHT - h)
                wall = pygame.Rect(x, y, w, h)
                if not wall.collidepoint(self.player_pos) and not wall.colliderect(self.exit_rect):
                    self.walls.append(wall)
                    break
        
        # --- Place Orbs and Obstacles ---
        for item_list, count, size in [(self.orbs, num_orbs, self.ORB_RADIUS), (self.obstacles, num_obstacles, 30)]:
            for _ in range(count):
                while True:
                    pos = pygame.Vector2(
                        self.np_random.integers(size, self.SCREEN_WIDTH - size),
                        self.np_random.integers(size, self.SCREEN_HEIGHT - size)
                    )
                    # Check for overlap with walls, start, and exit
                    is_colliding = any(w.collidepoint(pos) for w in self.walls)
                    if not is_colliding and pos.distance_to(self.player_pos) > 50 and self.exit_rect.collidepoint(pos) == 0:
                        if item_list is self.orbs:
                            item_list.append({'pos': pos, 'radius': self.ORB_RADIUS, 'pulse_phase': self.np_random.random() * math.pi})
                        elif item_list is self.obstacles:
                            axis = self.np_random.choice(['x', 'y'])
                            amplitude = self.np_random.integers(50, 150)
                            rect = pygame.Rect(0, 0, self.np_random.integers(20, 40), self.np_random.integers(20, 40))
                            rect.center = pos
                            item_list.append({
                                'rect': rect,
                                'base_pos': pos.copy(),
                                'axis': axis,
                                'amplitude': amplitude,
                                'frequency': 0.01 + obstacle_speed * self.np_random.random()
                            })
                        break

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "best_level": self.best_level,
            "timer": self.timer,
            "speed_multiplier": self.speed_multiplier
        }

    def _render_game(self):
        self._render_walls()
        self._render_exit()
        self._render_orbs()
        self._render_obstacles()
        self._render_particles()
        if not self.game_over:
            self._render_player()

    def _render_glow(self, pos, color, base_radius, num_layers=5, intensity=1.5):
        for i in range(num_layers, 0, -1):
            radius = int(base_radius + (i * intensity))
            alpha = int(255 / (i * 2 + 1))
            glow_color = (*color, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, glow_color)

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall, border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_WALL_GLOW, wall, width=2, border_radius=5)

    def _render_exit(self):
        self._render_glow(self.exit_rect.center, self.COLOR_EXIT_GLOW, self.exit_rect.width / 2, intensity=2.0)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect, border_radius=8)

    def _render_orbs(self):
        for orb in self.orbs:
            pulse = 1 + 0.2 * math.sin(self.steps * 0.1 + orb['pulse_phase'])
            radius = int(orb['radius'] * pulse)
            self._render_glow(orb['pos'], self.COLOR_ORB_GLOW, radius, intensity=1.8)
            pygame.gfxdraw.filled_circle(self.screen, int(orb['pos'].x), int(orb['pos'].y), radius, self.COLOR_ORB)

    def _render_obstacles(self):
        for obstacle in self.obstacles:
            self._render_glow(obstacle['rect'].center, self.COLOR_OBSTACLE_GLOW, obstacle['rect'].width / 1.5, intensity=2.5)
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obstacle['rect'], border_radius=3)
    
    def _render_player(self):
        self._render_glow(self.player_pos, self.COLOR_PLAYER_GLOW, self.PLAYER_RADIUS, intensity=2.0)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                'life': self.PARTICLE_LIFETIME,
                'color': color
            })

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / self.PARTICLE_LIFETIME))
            color = (*p['color'], alpha)
            size = int(5 * (p['life'] / self.PARTICLE_LIFETIME))
            if size > 0:
                rect = pygame.Rect(p['pos'].x - size/2, p['pos'].y - size/2, size, size)
                pygame.draw.rect(self.screen, color, rect)

    def _render_ui(self):
        # Level Text
        level_text = self.font_small.render(f"Level: {self.level}/{self.MAX_LEVELS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (10, 10))

        # Timer Text
        timer_text = self.font_large.render(f"{self.timer:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Speed Multiplier Text
        speed_text = self.font_small.render(f"Speed: {self.speed_multiplier:.2f}x", True, self.COLOR_UI_TEXT)
        text_rect = speed_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(speed_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # We need a real display for the manual test
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Maze Runner")
    
    done = False
    running = True
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while running:
        action_movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action_movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action_movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action_movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action_movement = 4
            
        action_space = 0 if not keys[pygame.K_SPACE] else 1
        action_shift = 0 if not keys[pygame.K_LSHIFT] else 1

        action = [action_movement, action_space, action_shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0
                done = False

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        
        if done:
            font = pygame.font.Font(None, 50)
            text = font.render("GAME OVER. Press 'R' to restart.", True, (255, 255, 255))
            text_rect = text.get_rect(center=(screen_width/2, screen_height/2))
            display_screen.blit(text, text_rect)
            
        pygame.display.flip()
        clock.tick(env.LOGIC_FPS)

    env.close()