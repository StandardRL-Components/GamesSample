
# Generated: 2025-08-27T19:28:50.772775
# Source Brief: brief_02168.md
# Brief Index: 2168

        
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move your robot. Avoid red obstacles and reach the green goal."
    )

    # User-facing game description
    game_description = (
        "Guide a robot through obstacle-laden courses to reach the goal as fast as possible. "
        "Collect yellow stars for points and to clear nearby obstacles."
    )

    # Frames auto-advance for smooth graphics and time-based gameplay
    auto_advance = True

    # Class-level counter for difficulty scaling across episodes
    total_steps_completed = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 30
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_GLOW = (60, 160, 255, 50)
        self.COLOR_OBSTACLE = (255, 70, 70)
        self.COLOR_OBSTACLE_GLOW = (255, 70, 70, 60)
        self.COLOR_GOAL = (70, 255, 70)
        self.COLOR_GOAL_GLOW = (70, 255, 70, 60)
        self.COLOR_BONUS = (255, 220, 50)
        self.COLOR_BONUS_GLOW = (255, 220, 50, 70)
        self.COLOR_WALL = (200, 200, 220)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_SHADOW = (10, 10, 15)
        
        # --- Entity Properties ---
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 5.0
        self.OBSTACLE_RADIUS = 8
        self.BONUS_RADIUS = 10
        self.GOAL_SIZE = 24
        self.BONUS_CLEAR_RADIUS = 80
        self.INITIAL_OBSTACLE_COUNT = 10
        self.BONUS_COUNT = 3

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Game State Initialization ---
        self.player_pos = pygame.Vector2(0, 0)
        self.goal_rect = pygame.Rect(0, 0, 0, 0)
        self.obstacles = []
        self.bonuses = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False
        self.game_outcome = ""
        self.rng = None

        # Initialize state variables
        self.reset()
        
        # self.validate_implementation() # Optional: Call to check compliance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.game_outcome = ""
        self.particles.clear()

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        
        # --- Game Logic ---
        reward = 0
        self.steps += 1
        GameEnv.total_steps_completed += 1
        self.time_remaining -= 1

        old_dist_to_goal = self.player_pos.distance_to(self.goal_rect.center)
        
        # --- Player Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            move_vec.y = -1
        elif movement == 2:  # Down
            move_vec.y = 1
        elif movement == 3:  # Left
            move_vec.x = -1
        elif movement == 4:  # Right
            move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            # Player trail particles
            if self.steps % 2 == 0:
                self._create_particles(self.player_pos, 1, self.COLOR_PLAYER, 1.5, 10)

        # --- Boundary Collision ---
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)
        
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # --- Reward for distance change ---
        new_dist_to_goal = self.player_pos.distance_to(self.goal_rect.center)
        if new_dist_to_goal < old_dist_to_goal:
            reward += 0.1
        else:
            reward -= 0.2

        # --- Check Collisions ---
        # Goal
        if player_rect.colliderect(self.goal_rect):
            # sfx: goal_reached.wav
            self.game_over = True
            self.game_outcome = "GOAL REACHED!"
            reward += 50
            self.score += 1000
            self._create_particles(self.player_pos, 50, self.COLOR_GOAL, 4, 40)

        # Bonuses
        for bonus_pos in self.bonuses[:]:
            if self.player_pos.distance_to(bonus_pos) < self.PLAYER_SIZE / 2 + self.BONUS_RADIUS:
                # sfx: bonus_collect.wav
                self.bonuses.remove(bonus_pos)
                reward += 5
                self.score += 100
                self._create_particles(bonus_pos, 30, self.COLOR_BONUS, 3, 30)
                
                # Clear nearby obstacles
                obstacles_to_keep = []
                for obs_pos in self.obstacles:
                    if obs_pos.distance_to(bonus_pos) > self.BONUS_CLEAR_RADIUS:
                        obstacles_to_keep.append(obs_pos)
                    else:
                        # sfx: obstacle_destroy.wav
                        self._create_particles(obs_pos, 20, self.COLOR_OBSTACLE, 2.5, 25)
                self.obstacles = obstacles_to_keep

        # Obstacles
        for obs_pos in self.obstacles:
            if self.player_pos.distance_to(obs_pos) < self.PLAYER_SIZE / 2 + self.OBSTACLE_RADIUS:
                # sfx: player_hit.wav
                self.game_over = True
                self.game_outcome = "OBSTACLE HIT!"
                reward = -100  # Per brief, this is the final reward
                self._create_particles(self.player_pos, 50, self.COLOR_OBSTACLE, 5, 40)
                break
        
        # --- Update Particles ---
        self._update_particles()
        
        # --- Termination Check ---
        terminated = self.game_over
        if self.time_remaining <= 0 and not terminated:
            # sfx: time_up.wav
            self.game_over = True
            terminated = True
            self.game_outcome = "TIME'S UP!"
            reward -= 10 # Small penalty for timeout
        
        if self.steps >= self.MAX_STEPS and not terminated:
            terminated = True
            self.game_over = True
            self.game_outcome = "MAX STEPS REACHED"

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_level(self):
        # Place player on the left, goal on the right
        self.player_pos = pygame.Vector2(
            self.rng.uniform(self.WIDTH * 0.1, self.WIDTH * 0.2),
            self.rng.uniform(self.HEIGHT * 0.1, self.HEIGHT * 0.9)
        )
        goal_x = self.rng.uniform(self.WIDTH * 0.8, self.WIDTH * 0.9)
        goal_y = self.rng.uniform(self.HEIGHT * 0.1, self.HEIGHT * 0.9)
        self.goal_rect = pygame.Rect(goal_x - self.GOAL_SIZE / 2, goal_y - self.GOAL_SIZE / 2, self.GOAL_SIZE, self.GOAL_SIZE)

        # Determine obstacle count based on total steps
        difficulty_tier = GameEnv.total_steps_completed // 500
        num_obstacles = self.INITIAL_OBSTACLE_COUNT + difficulty_tier

        # Generate obstacles and bonuses
        self.obstacles = self._generate_positions(num_obstacles, self.OBSTACLE_RADIUS)
        self.bonuses = self._generate_positions(self.BONUS_COUNT, self.BONUS_RADIUS)
    
    def _generate_positions(self, count, radius):
        positions = []
        occupied_rects = [
            pygame.Rect(self.player_pos.x - 50, self.player_pos.y - 50, 100, 100),
            pygame.Rect(self.goal_rect.x - 50, self.goal_rect.y - 50, 100, 100)
        ]
        
        for _ in range(count):
            for _ in range(100): # Max 100 attempts to place an item
                pos = pygame.Vector2(
                    self.rng.uniform(radius, self.WIDTH - radius),
                    self.rng.uniform(radius, self.HEIGHT - radius)
                )
                
                # Check for overlap with other generated items and key areas
                is_overlapping = False
                new_rect = pygame.Rect(pos.x - radius, pos.y - radius, radius * 2, radius * 2)
                
                for r in occupied_rects:
                    if r.colliderect(new_rect):
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    positions.append(pos)
                    occupied_rects.append(new_rect)
                    break
        return positions

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "time_remaining": self.time_remaining}
    
    def _render_game(self):
        # Draw shadows first
        shadow_offset = 3
        pygame.draw.rect(self.screen, self.COLOR_SHADOW, self.goal_rect.move(shadow_offset, shadow_offset))
        for obs_pos in self.obstacles:
            pygame.draw.circle(self.screen, self.COLOR_SHADOW, (int(obs_pos.x + shadow_offset), int(obs_pos.y + shadow_offset)), self.OBSTACLE_RADIUS)
        player_rect_shadow = pygame.Rect(0,0,self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect_shadow.center = self.player_pos + pygame.Vector2(shadow_offset, shadow_offset)
        pygame.draw.rect(self.screen, self.COLOR_SHADOW, player_rect_shadow)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            p['color'] = p['base_color'][:3] + (alpha,)
            self._draw_aa_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'])

        # Draw Goal
        self._draw_glowing_rect(self.screen, self.goal_rect, self.COLOR_GOAL, self.COLOR_GOAL_GLOW)
        
        # Draw Bonuses
        for bonus_pos in self.bonuses:
            self._draw_glowing_star(self.screen, bonus_pos, self.BONUS_RADIUS, self.COLOR_BONUS, self.COLOR_BONUS_GLOW)

        # Draw Obstacles
        for obs_pos in self.obstacles:
            self._draw_glowing_circle(self.screen, obs_pos, self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_GLOW)

        # Draw Player
        player_rect = pygame.Rect(0,0,self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        self._draw_glowing_rect(self.screen, player_rect, self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Time remaining
        time_text = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH - score_surf.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            end_text_surf = self.font_game_over.render(self.game_outcome, True, self.COLOR_TEXT)
            text_rect = end_text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text_surf, text_rect)

    # --- Particle System ---
    def _create_particles(self, pos, count, color, speed_multiplier, life):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(0.5, 1.5) * speed_multiplier
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.rng.integers(life // 2, life),
                'max_life': life,
                'size': self.rng.uniform(1, 4),
                'base_color': color
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95  # friction
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    # --- Drawing Helpers for Visual Polish ---
    def _draw_aa_circle(self, surface, x, y, radius, color):
        pygame.gfxdraw.aacircle(surface, x, y, radius, color)
        pygame.gfxdraw.filled_circle(surface, x, y, radius, color)

    def _draw_glowing_circle(self, surface, pos, radius, color, glow_color):
        self._draw_aa_circle(surface, int(pos.x), int(pos.y), int(radius * 1.8), glow_color)
        self._draw_aa_circle(surface, int(pos.x), int(pos.y), radius, color)

    def _draw_glowing_rect(self, surface, rect, color, glow_color):
        glow_rect = rect.inflate(rect.width * 0.8, rect.height * 0.8)
        # This is a trick to draw a soft-edged rect
        pygame.draw.rect(surface, glow_color, glow_rect, border_radius=int(glow_rect.width*0.3))
        pygame.draw.rect(surface, color, rect, border_radius=int(rect.width*0.3))
        
    def _draw_glowing_star(self, surface, center_pos, radius, color, glow_color):
        points = self._get_star_points(center_pos, radius)
        glow_points = self._get_star_points(center_pos, radius * 1.8)
        
        pygame.gfxdraw.aapolygon(surface, glow_points, glow_color)
        pygame.gfxdraw.filled_polygon(surface, glow_points, glow_color)
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _get_star_points(self, center, radius, num_points=5):
        points = []
        for i in range(num_points * 2):
            r = radius if i % 2 == 0 else radius * 0.4
            angle = i * math.pi / num_points - math.pi / 2
            points.append(
                (center.x + r * math.cos(angle), center.y + r * math.sin(angle))
            )
        return points

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_window = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Maze Runner")
    clock = pygame.time.Clock()

    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the game window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        game_window.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
    
    env.close()