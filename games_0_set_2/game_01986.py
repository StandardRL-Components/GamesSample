
# Generated: 2025-08-27T18:54:46.063201
# Source Brief: brief_01986.md
# Brief Index: 1986

        
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
        "Controls: ↑ to move the cart up, ↓ to move down. Avoid obstacles and reach the exit!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated mine shaft in a speeding cart. "
        "Dodge stalactites and stalagmites to reach the end before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 30
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.LEVEL_LENGTH_PIXELS = 15000 # Total length of the mine shaft

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_SHAFT_BG = (40, 40, 50)
        self.COLOR_PLAYER = (220, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 50)
        self.COLOR_OBSTACLE = (90, 60, 40)
        self.COLOR_OBSTACLE_EDGE = (120, 90, 70)
        self.COLOR_GOAL = (255, 220, 0)
        self.COLOR_GOAL_GLOW = (255, 220, 0, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_RAIL = (70, 70, 80)
        self.COLOR_BEAM = (60, 45, 30)

        # Player physics
        self.PLAYER_SPEED_VERTICAL = 6
        self.PLAYER_WIDTH = 40
        self.PLAYER_HEIGHT = 30
        self.PLAYER_INITIAL_X = self.SCREEN_WIDTH // 4

        # World physics
        self.HORIZONTAL_SPEED = 8
        self.SHAFT_TOP_MARGIN = 50
        self.SHAFT_BOTTOM_MARGIN = 50

        # Obstacle generation
        self.INITIAL_OBSTACLE_DENSITY = 0.02
        self.OBSTACLE_DENSITY_INCREASE = 0.05
        self.MIN_OBSTACLE_GAP = self.PLAYER_HEIGHT * 2.5
        self.OBSTACLE_MIN_WIDTH = 20
        self.OBSTACLE_MAX_WIDTH = 80
        self.OBSTACLE_SPACING = 250

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.game_font_small = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.game_font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
        except pygame.error:
            self.game_font_small = pygame.font.SysFont("monospace", 18)
            self.game_font_large = pygame.font.SysFont("monospace", 24)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_rect = None
        self.camera_x = 0
        self.obstacles = []
        self.passed_obstacles = set()
        self.particles = []
        self.obstacle_density = self.INITIAL_OBSTACLE_DENSITY
        self.generation_frontier_x = 0
        self.background_beams = []
        
        # Initialize state and validate
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.camera_x = 0
        self.generation_frontier_x = 0
        
        player_y = self.SCREEN_HEIGHT / 2
        self.player_rect = pygame.Rect(
            self.PLAYER_INITIAL_X, player_y - self.PLAYER_HEIGHT / 2,
            self.PLAYER_WIDTH, self.PLAYER_HEIGHT
        )

        self.obstacles = []
        self.passed_obstacles = set()
        self.particles = []
        self.background_beams = []
        
        self.obstacle_density = self.INITIAL_OBSTACLE_DENSITY
        
        self._generate_world_elements(self.LEVEL_LENGTH_PIXELS + self.SCREEN_WIDTH)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)
            
        reward = 0
        terminated = self.game_over

        if not terminated:
            # --- Action Handling ---
            movement = action[0]
            if movement == 1:  # Up
                self.player_rect.y -= self.PLAYER_SPEED_VERTICAL
            elif movement == 2:  # Down
                self.player_rect.y += self.PLAYER_SPEED_VERTICAL

            # --- Game Logic ---
            # Player bounds
            self.player_rect.y = np.clip(
                self.player_rect.y,
                self.SHAFT_TOP_MARGIN,
                self.SCREEN_HEIGHT - self.SHAFT_BOTTOM_MARGIN - self.PLAYER_HEIGHT
            )
            
            # World scrolling
            self.camera_x += self.HORIZONTAL_SPEED

            # --- Update State & Rewards ---
            # Survival reward
            reward += 0.1

            # Proximity risk penalty
            player_risk_rect = self.player_rect.inflate(20, 20)
            for i, obs_list in enumerate(self.obstacles):
                # Only check obstacles that are on screen
                if obs_list[0].right > self.camera_x and obs_list[0].left < self.camera_x + self.SCREEN_WIDTH:
                    for obs in obs_list:
                        if player_risk_rect.colliderect(obs.move(-self.camera_x, 0)):
                            reward -= 0.2
                            break
            
            # Check for passing obstacles
            for i, obs_list in enumerate(self.obstacles):
                if i not in self.passed_obstacles:
                    if obs_list[0].right < self.camera_x + self.player_rect.left:
                        reward += 10.0
                        self.passed_obstacles.add(i)
                        # sound: 'ding'

            # --- Termination Checks ---
            # Collision
            for obs_list in self.obstacles:
                if self.player_rect.collidelist([obs.move(-self.camera_x, 0) for obs in obs_list]) != -1:
                    terminated = True
                    reward = -100.0
                    self._create_explosion(self.player_rect.center)
                    # sound: 'explosion'
                    break
            
            if not terminated:
                # Win condition
                if self.camera_x >= self.LEVEL_LENGTH_PIXELS:
                    terminated = True
                    reward = 100.0
                    # sound: 'victory'
                
                # Time limit
                elif self.steps >= self.MAX_STEPS:
                    terminated = True
                    reward = -50.0
                    # sound: 'timeout'

        # --- Final Updates ---
        self.steps += 1
        self.score += reward
        self.game_over = terminated
        
        self._update_particles()
        
        if self.steps > 0 and self.steps % 1000 == 0:
            self.obstacle_density *= (1 + self.OBSTACLE_DENSITY_INCREASE)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _generate_world_elements(self, until_x):
        beam_spacing = 400
        while self.generation_frontier_x < until_x:
            # Generate Beams
            if self.generation_frontier_x > len(self.background_beams) * beam_spacing:
                 self.background_beams.append(len(self.background_beams) * beam_spacing)
            
            # Generate Obstacles
            if self.generation_frontier_x > self.SCREEN_WIDTH and self.np_random.random() < self.obstacle_density:
                gap_y = self.np_random.integers(
                    self.SHAFT_TOP_MARGIN + self.MIN_OBSTACLE_GAP / 2,
                    self.SCREEN_HEIGHT - self.SHAFT_BOTTOM_MARGIN - self.MIN_OBSTACLE_GAP / 2
                )
                width = self.np_random.integers(self.OBSTACLE_MIN_WIDTH, self.OBSTACLE_MAX_WIDTH)
                
                obs_top_height = gap_y - self.MIN_OBSTACLE_GAP / 2 - self.SHAFT_TOP_MARGIN
                rect_top = pygame.Rect(
                    self.generation_frontier_x, self.SHAFT_TOP_MARGIN,
                    width, obs_top_height
                )
                
                obs_bottom_y = gap_y + self.MIN_OBSTACLE_GAP / 2
                obs_bottom_height = (self.SCREEN_HEIGHT - self.SHAFT_BOTTOM_MARGIN) - obs_bottom_y
                rect_bottom = pygame.Rect(
                    self.generation_frontier_x, obs_bottom_y,
                    width, obs_bottom_height
                )
                
                # Store obstacles with world coordinates
                if rect_top.height > 5 and rect_bottom.height > 5:
                    self.obstacles.append([rect_top, rect_bottom])

            self.generation_frontier_x += self.OBSTACLE_SPACING

    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            size = self.np_random.integers(3, 7)
            color = random.choice([(255, 180, 0), (255, 100, 0), (200, 200, 200)])
            particle = {
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "size": size,
                "color": color,
                "life": self.np_random.integers(15, 30)
            }
            self.particles.append(particle)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1  # gravity
            p["life"] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        shaft_rect = pygame.Rect(0, self.SHAFT_TOP_MARGIN, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.SHAFT_TOP_MARGIN - self.SHAFT_BOTTOM_MARGIN)
        pygame.draw.rect(self.screen, self.COLOR_SHAFT_BG, shaft_rect)

        for beam_x in self.background_beams:
            screen_x = int(beam_x - self.camera_x)
            if -40 < screen_x < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_BEAM, (screen_x, self.SHAFT_TOP_MARGIN, 20, shaft_rect.height))

        rail_y = self.player_rect.bottom + 4
        for i in range(-int(self.camera_x % 20), self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_RAIL, (i, rail_y), (i+10, rail_y), 2)
            pygame.draw.line(self.screen, self.COLOR_RAIL, (i, rail_y + 4), (i+10, rail_y + 4), 2)

        goal_screen_x = self.LEVEL_LENGTH_PIXELS - self.camera_x
        if goal_screen_x < self.SCREEN_WIDTH:
            goal_rect = pygame.Rect(goal_screen_x, self.SHAFT_TOP_MARGIN, 50, shaft_rect.height)
            pygame.gfxdraw.box(self.screen, goal_rect, self.COLOR_GOAL)
            pygame.gfxdraw.box(self.screen, goal_rect.inflate(10, 10), self.COLOR_GOAL_GLOW)

        for obs_list in self.obstacles:
            for obs in obs_list:
                screen_obs = obs.move(-self.camera_x, 0)
                if screen_obs.right > 0 and screen_obs.left < self.SCREEN_WIDTH:
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_obs)
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_EDGE, screen_obs, 2)
        
        if not self.game_over or len(self.particles) > 0:
            glow_surface = pygame.Surface((self.PLAYER_WIDTH * 2, self.PLAYER_HEIGHT * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect())
            self.screen.blit(glow_surface, (self.player_rect.centerx - self.PLAYER_WIDTH, self.player_rect.centery - self.PLAYER_HEIGHT), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)
            pygame.draw.rect(self.screen, (0,0,0), self.player_rect.inflate(-4,-4))

            wheel_radius = 5
            wheel_y = self.player_rect.bottom
            wheel_x1 = self.player_rect.left + 8
            wheel_x2 = self.player_rect.right - 8
            pygame.draw.circle(self.screen, (100,100,100), (wheel_x1, wheel_y), wheel_radius)
            pygame.draw.circle(self.screen, (100,100,100), (wheel_x2, wheel_y), wheel_radius)
            
            angle = (self.camera_x * 0.1) % (2 * math.pi)
            pygame.draw.line(self.screen, (50,50,50), (wheel_x1, wheel_y), (wheel_x1 + math.cos(angle)*wheel_radius, wheel_y + math.sin(angle)*wheel_radius), 2)
            pygame.draw.line(self.screen, (50,50,50), (wheel_x2, wheel_y), (wheel_x2 + math.cos(angle)*wheel_radius, wheel_y + math.sin(angle)*wheel_radius), 2)

        for p in self.particles:
            pos = (int(p["pos"][0] - self.camera_x), int(p["pos"][1]))
            pygame.draw.rect(self.screen, p["color"], (*pos, p["size"], p["size"]))
            
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, 0, self.SCREEN_WIDTH, self.SHAFT_TOP_MARGIN))
        pygame.draw.rect(self.screen, self.COLOR_BG, (0, self.SCREEN_HEIGHT - self.SHAFT_BOTTOM_MARGIN, self.SCREEN_WIDTH, self.SHAFT_BOTTOM_MARGIN))

    def _render_ui(self):
        score_text = self.game_font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        time_text = self.game_font_large.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)
        
        progress = self.camera_x / self.LEVEL_LENGTH_PIXELS
        bar_width = self.SCREEN_WIDTH - 20
        bar_height = 10
        pygame.draw.rect(self.screen, (50,50,50), (10, self.SCREEN_HEIGHT - 20, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_GOAL, (10, self.SCREEN_HEIGHT - 20, bar_width * progress, bar_height))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS)),
            "progress": self.camera_x / self.LEVEL_LENGTH_PIXELS
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Mine Cart Madness")
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        movement = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            movement = 2
        
        action[0] = movement
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
    env.close()