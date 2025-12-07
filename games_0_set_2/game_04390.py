
# Generated: 2025-08-28T02:15:11.843796
# Source Brief: brief_04390.md
# Brief Index: 4390

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    An arcade racing game where the player must jump over procedurally generated
    obstacles to reach the finish line against a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Press SPACE to jump over the red obstacles."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced retro-futuristic side-scroller. Jump over obstacles to reach the finish line before time runs out."
    )

    # Frames auto-advance for smooth, time-based gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60 # Visual rendering FPS
    
    # Colors
    COLOR_BG = (10, 0, 20)
    COLOR_TRACK = (40, 20, 60)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_PLAYER_GLOW = (0, 100, 150)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_FINISH_LINE = (50, 255, 50)
    COLOR_TEXT = (255, 255, 255)
    
    # Player Physics
    PLAYER_X_POS = 120
    PLAYER_BASE_WIDTH, PLAYER_BASE_HEIGHT = 20, 40
    GRAVITY = 1.0
    JUMP_STRENGTH = -16
    
    # Game Rules
    TRACK_Y = HEIGHT - 50
    MAX_STEPS = 900 # 30 seconds at 30 steps/sec
    MAX_COLLISIONS = 5
    FINISH_LINE_POS = 15000

    # Difficulty Scaling
    INITIAL_SCROLL_SPEED = 12
    SPEED_INCREASE_INTERVAL = 300 # steps
    SPEED_INCREASE_AMOUNT = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_big = pygame.font.Font(None, 72)
        
        self.render_mode = render_mode
        self.np_random = None

        # These attributes are reset in `reset()`
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_y = 0
        self.player_vy = 0
        self.player_width = self.PLAYER_BASE_WIDTH
        self.player_height = self.PLAYER_BASE_HEIGHT
        self.squash_stretch_factor = 0
        self.on_ground = False
        self.world_progress = 0
        self.scroll_speed = 0
        self.collision_count = 0
        self.obstacles = []
        self.particles = []
        self.parallax_stars = []
        self.next_obstacle_spawn_dist = 0
        self.obstacle_id_counter = 0
        self.collided_obstacle_ids = set()
        
        self.validate_implementation()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_y = self.TRACK_Y - self.PLAYER_BASE_HEIGHT
        self.player_vy = 0
        self.player_width = self.PLAYER_BASE_WIDTH
        self.player_height = self.PLAYER_BASE_HEIGHT
        self.squash_stretch_factor = 0
        self.on_ground = True
        
        self.world_progress = 0
        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.collision_count = 0
        
        self.obstacles = []
        self.particles = []
        self.collided_obstacle_ids = set()
        self.obstacle_id_counter = 0
        
        self._generate_initial_stars()
        self.next_obstacle_spawn_dist = self.WIDTH * 1.5

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        space_held = action[1] == 1
        self.steps += 1
        
        # --- Game Logic Update ---
        self._handle_input(space_held)
        self._update_player()
        self._update_world()
        self._update_obstacles()
        self._update_particles()
        
        # --- Collision and Rewards ---
        collision_this_frame = self._check_collisions()
        reward = self._calculate_reward(collision_this_frame)
        self.score += reward
        
        # --- Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.world_progress >= self.FINISH_LINE_POS:
                time_bonus = 50 * (self.MAX_STEPS - self.steps) / self.MAX_STEPS
                self.score += time_bonus
                reward += time_bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, space_held):
        if space_held and self.on_ground:
            self.player_vy = self.JUMP_STRENGTH
            self.on_ground = False
            self.squash_stretch_factor = 1.0 # Start stretch for jump
            # SFX: Jump sound
            self._create_particles(self.PLAYER_X_POS + self.PLAYER_BASE_WIDTH / 2, self.TRACK_Y, 20, self.COLOR_PLAYER)

    def _update_player(self):
        # Apply gravity
        self.player_vy += self.GRAVITY
        self.player_y += self.player_vy

        # Ground check
        if self.player_y >= self.TRACK_Y - self.PLAYER_BASE_HEIGHT:
            if not self.on_ground:
                # Landed
                self.squash_stretch_factor = -1.0 # Start squash for landing
                self._create_particles(self.PLAYER_X_POS + self.PLAYER_BASE_WIDTH / 2, self.TRACK_Y, 10, self.COLOR_PLAYER)
                # SFX: Land sound
            self.player_y = self.TRACK_Y - self.PLAYER_BASE_HEIGHT
            self.player_vy = 0
            self.on_ground = True
        else:
            self.on_ground = False
            
        # Update squash and stretch
        if self.squash_stretch_factor != 0:
            self.squash_stretch_factor *= 0.85 # Dampen effect
            if abs(self.squash_stretch_factor) < 0.05:
                self.squash_stretch_factor = 0
        
        stretch = self.squash_stretch_factor * 0.4
        self.player_height = self.PLAYER_BASE_HEIGHT * (1 + stretch)
        self.player_width = self.PLAYER_BASE_WIDTH * (1 - stretch)

    def _update_world(self):
        self.world_progress += self.scroll_speed

        # Difficulty scaling
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.scroll_speed += self.SPEED_INCREASE_AMOUNT

    def _update_obstacles(self):
        # Move existing obstacles and remove off-screen ones
        for obstacle in self.obstacles[:]:
            obstacle['x'] -= self.scroll_speed
            if obstacle['x'] + obstacle['w'] < 0:
                self.obstacles.remove(obstacle)

        # Spawn new obstacles
        if self.world_progress > self.next_obstacle_spawn_dist:
            difficulty_factor = 1 - (self.steps / (self.MAX_STEPS * 1.5))
            min_gap = max(200, 400 * difficulty_factor)
            max_gap = max(400, 800 * difficulty_factor)
            
            self.next_obstacle_spawn_dist += self.np_random.integers(min_gap, max_gap)
            
            height = self.np_random.integers(20, 60)
            width = self.np_random.integers(20, 40)
            
            self.obstacles.append({
                'id': self.obstacle_id_counter,
                'x': self.WIDTH + 50,
                'y': self.TRACK_Y - height,
                'w': width,
                'h': height,
                'cleared': False
            })
            self.obstacle_id_counter += 1

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] - 0.2)
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_collisions(self):
        player_rect = pygame.Rect(
            self.PLAYER_X_POS, self.player_y, self.player_width, self.player_height
        )
        collision_this_frame = False
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h'])
            if player_rect.colliderect(obstacle_rect):
                if obstacle['id'] not in self.collided_obstacle_ids:
                    self.collision_count += 1
                    self.collided_obstacle_ids.add(obstacle['id'])
                    collision_this_frame = True
                    self._create_particles(player_rect.centerx, player_rect.centery, 30, self.COLOR_OBSTACLE)
                    # SFX: Collision/hit sound
        return collision_this_frame

    def _calculate_reward(self, collision_this_frame):
        reward = 0.1 # Base reward for surviving a step

        if collision_this_frame:
            reward -= 5.0

        player_left_edge = self.PLAYER_X_POS
        for obstacle in self.obstacles:
            if not obstacle['cleared'] and obstacle['x'] + obstacle['w'] < player_left_edge:
                if obstacle['id'] not in self.collided_obstacle_ids:
                    reward += 5.0 # Reward for successfully clearing an obstacle
                obstacle['cleared'] = True
        
        return reward

    def _check_termination(self):
        return (
            self.collision_count >= self.MAX_COLLISIONS or
            self.steps >= self.MAX_STEPS or
            self.world_progress >= self.FINISH_LINE_POS
        )

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collisions": self.collision_count,
            "progress": self.world_progress,
        }
        
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        self._render_parallax_stars()

        # Track
        pygame.draw.rect(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y, self.WIDTH, self.HEIGHT - self.TRACK_Y))

        # Finish Line
        finish_line_screen_x = self.FINISH_LINE_POS - self.world_progress
        if 0 < finish_line_screen_x < self.WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_line_screen_x, 0), (finish_line_screen_x, self.HEIGHT), 5)

        # Obstacles
        for obstacle in self.obstacles:
            rect = pygame.Rect(int(obstacle['x']), int(obstacle['y']), int(obstacle['w']), int(obstacle['h']))
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)

        # Player
        player_rect = pygame.Rect(
            int(self.PLAYER_X_POS), int(self.player_y), int(self.player_width), int(self.player_height)
        )
        glow_radius = int(self.PLAYER_BASE_HEIGHT * 0.8)
        pygame.gfxdraw.filled_circle(
            self.screen, player_rect.centerx, player_rect.centery, glow_radius, (*self.COLOR_PLAYER_GLOW, 80)
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), (*p['color'], p['life'] * 5))

        # UI
        self._render_ui()

    def _render_ui(self):
        # Collision Counter
        collision_text = self.font_ui.render(f"Hits: {self.collision_count}/{self.MAX_COLLISIONS}", True, self.COLOR_TEXT)
        self.screen.blit(collision_text, (10, 10))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / 30.0
        timer_text = self.font_ui.render(f"Time: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = ""
            if self.world_progress >= self.FINISH_LINE_POS:
                message = "FINISH!"
            elif self.collision_count >= self.MAX_COLLISIONS:
                message = "TOO MANY HITS"
            elif self.steps >= self.MAX_STEPS:
                message = "TIME UP"
            
            game_over_text = self.font_big.render(message, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(game_over_text, text_rect)
            
            score_text = self.font_ui.render(f"Final Score: {self.score:.0f}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 30))
            self.screen.blit(score_text, score_rect)

    def _generate_initial_stars(self):
        self.parallax_stars = []
        for i in range(3): # 3 layers
            speed_multiplier = (i + 1) * 0.2
            num_stars = 50 * (3 - i)
            layer = []
            for _ in range(num_stars):
                layer.append({
                    'x': self.np_random.integers(0, self.WIDTH),
                    'y': self.np_random.integers(0, self.TRACK_Y),
                    'size': self.np_random.integers(1, 4-i),
                    'speed': speed_multiplier
                })
            self.parallax_stars.append(layer)

    def _render_parallax_stars(self):
        for layer in self.parallax_stars:
            for star in layer:
                star['x'] -= self.scroll_speed * star['speed']
                if star['x'] < 0:
                    star['x'] = self.WIDTH
                    star['y'] = self.np_random.integers(0, self.TRACK_Y)
                
                color_val = int(100 * star['speed'])
                color = (color_val, color_val, color_val + 50)
                pygame.draw.rect(self.screen, color, (int(star['x']), int(star['y']), star['size'], star['size']))

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': self.np_random.integers(3, 7),
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Temporarily set up minimal state for one-time observation generation
        self._generate_initial_stars()
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    done = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        action = [0, 0, 0] # No-op
        if keys[pygame.K_SPACE]:
            action[1] = 1 # Jump
            
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                done = False
                total_reward = 0

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(GameEnv.FPS)

    env.close()