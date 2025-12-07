
# Generated: 2025-08-28T03:35:55.721099
# Source Brief: brief_04974.md
# Brief Index: 4974

        
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
        "Controls: Press space to jump over obstacles. Your robot runs automatically."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Guide a robot across a procedurally generated landscape. "
        "Time your jumps to clear obstacles and reach 1000 meters."
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        # Colors
        self.COLOR_SKY = (135, 206, 235)
        self.COLOR_GROUND = (139, 69, 19)
        self.COLOR_ROBOT = (0, 128, 255)
        self.COLOR_ROBOT_GLOW = (100, 180, 255)
        self.COLOR_OBSTACLE = (220, 20, 60)
        self.COLOR_OBSTACLE_GLOW = (255, 100, 120)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        self.PARALLAX_BG_COLOR = (100, 150, 180)

        # Game constants
        self.GROUND_Y = 350
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -12
        self.ROBOT_SPEED = 5  # pixels per step
        self.MAX_STEPS = 10000
        self.GOAL_DISTANCE_METERS = 1000

        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.goal_reached = False
        self.reward_this_step = 0
        self.robot_pos = pygame.Vector2(0, 0)
        self.robot_vel = pygame.Vector2(0, 0)
        self.robot_on_ground = True
        self.robot_width = 24
        self.robot_height = 36
        self.world_scroll = 0
        self.distance_traveled = 0
        self.obstacles = []
        self.next_obstacle_dist = 0
        self.particles = []
        self.parallax_hills = []
        self.difficulty_level = 0
        self.robot_screen_x = 100

        # Initialize state
        self.reset()
        
        # Run self-check
        # self.validate_implementation() # Commented out for submission, but good for local testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.goal_reached = False
        self.reward_this_step = 0

        # Robot state
        self.robot_pos = pygame.Vector2(self.robot_screen_x, self.GROUND_Y - self.robot_height)
        self.robot_vel = pygame.Vector2(0, 0)
        self.robot_on_ground = True

        # World state
        self.world_scroll = 0
        self.distance_traveled = 0
        self.obstacles = []
        self.particles = []
        self.difficulty_level = 0

        # Procedural generation init
        self.next_obstacle_dist = self.WIDTH * 0.8
        self._generate_initial_obstacles()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: unused
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        self.reward_this_step = 0

        if not self.game_over:
            # Update game logic
            self._handle_input(space_held)
            self._update_physics()
            self._update_world()
        
        self.steps += 1
        
        reward = self._calculate_reward()
        terminated = self._check_termination()

        if terminated:
            if self.goal_reached:
                reward += 100 # Goal reward
            elif self.game_over:
                reward += -10 # Collision penalty
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, space_held):
        if space_held and self.robot_on_ground:
            self.robot_vel.y = self.JUMP_STRENGTH
            self.robot_on_ground = False
            # sfx: Jump sound

    def _update_physics(self):
        # Update robot
        if not self.robot_on_ground:
            self.robot_vel.y += self.GRAVITY
        
        self.robot_pos.y += self.robot_vel.y

        # Ground collision
        landed = False
        if self.robot_pos.y >= self.GROUND_Y - self.robot_height:
            if not self.robot_on_ground:
                landed = True
            self.robot_pos.y = self.GROUND_Y - self.robot_height
            self.robot_vel.y = 0
            self.robot_on_ground = True
        
        if landed:
            self._create_landing_particles(5)
            # sfx: Land sound

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _update_world(self):
        # Scroll world
        self.world_scroll += self.ROBOT_SPEED
        self.distance_traveled += self.ROBOT_SPEED
        self.score = self.distance_traveled / 10 # 10 pixels = 1 meter
        
        # Update difficulty
        self.difficulty_level = int(self.score // 200)

        # Check for goal
        if not self.goal_reached and self.score >= self.GOAL_DISTANCE_METERS:
            self.goal_reached = True
            self.game_over = True # End game on win

        # Update obstacles
        robot_rect = self.get_robot_rect()
        for obs in self.obstacles[:]:
            obs['rect'].x -= self.ROBOT_SPEED
            
            # Check for passing obstacle
            if not obs['passed'] and obs['rect'].right < robot_rect.left:
                obs['passed'] = True
                clearance = robot_rect.bottom - obs['rect'].top
                if 0 <= clearance < 5:
                    self.reward_this_step += 2 # Risky jump reward
                    # sfx: Risky clear sound
            
            # Check for collision
            if robot_rect.colliderect(obs['rect']):
                self.game_over = True
                # sfx: Collision/Explosion sound

            # Remove off-screen obstacles
            if obs['rect'].right < 0:
                self.obstacles.remove(obs)
        
        # Generate new obstacles
        if len(self.obstacles) < 5:
            self._generate_obstacle()

    def _calculate_reward(self):
        reward = self.reward_this_step
        # Reward for travel
        reward += (self.ROBOT_SPEED / 10) * 0.1
        # Penalty for not jumping (applied per step)
        # Note: This is an arbitrary choice to encourage action.
        # Could be tied to being on the ground instead.
        if not self.robot_on_ground:
            reward -= 0.02
        return reward

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _generate_obstacle(self):
        last_obstacle_x = self.WIDTH
        if self.obstacles:
            last_obstacle_x = self.obstacles[-1]['rect'].right
        
        gap_reduction = self.difficulty_level * 5
        min_gap = max(120, 200 - gap_reduction)
        max_gap = max(200, 300 - gap_reduction)
        
        gap = self.np_random.integers(min_gap, max_gap)
        x_pos = last_obstacle_x + gap
        
        width = self.np_random.integers(20, 50)
        max_jump_height_pixels = 110 # Tuned based on physics
        height = self.np_random.integers(20, min(80, max_jump_height_pixels))

        obs_rect = pygame.Rect(x_pos, self.GROUND_Y - height, width, height)
        self.obstacles.append({'rect': obs_rect, 'passed': False})

    def _generate_initial_obstacles(self):
        self._generate_obstacle()
        while self.obstacles[-1]['rect'].x < self.WIDTH:
             self._generate_obstacle()
        
        # Generate parallax hills
        self.parallax_hills = []
        for i in range(20):
            x = self.np_random.integers(-self.WIDTH, self.WIDTH * 2)
            y = self.np_random.integers(self.GROUND_Y - 100, self.GROUND_Y - 20)
            radius = self.np_random.integers(50, 150)
            self.parallax_hills.append({'x': x, 'y': y, 'radius': radius})

    def get_robot_rect(self):
        return pygame.Rect(self.robot_pos.x, self.robot_pos.y, self.robot_width, self.robot_height)

    def _create_landing_particles(self, count):
        robot_rect = self.get_robot_rect()
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(robot_rect.centerx, robot_rect.bottom),
                'vel': pygame.Vector2(self.np_random.uniform(-1.5, 1.5), self.np_random.uniform(-2, -0.5)),
                'life': self.np_random.integers(15, 30),
                'color': (160, 82, 45)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_SKY)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw parallax background
        for hill in self.parallax_hills:
            scroll_x = int(hill['x'] - self.world_scroll * 0.3)
            pygame.gfxdraw.filled_circle(self.screen, scroll_x, hill['y'], hill['radius'], self.PARALLAX_BG_COLOR)
            pygame.gfxdraw.aacircle(self.screen, scroll_x, hill['y'], hill['radius'], self.PARALLAX_BG_COLOR)

        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Draw obstacles
        for obs in self.obstacles:
            if obs['rect'].right > 0 and obs['rect'].left < self.WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
                glow_rect = obs['rect'].inflate(4, 4)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, 2, border_radius=3)

        # Draw particles
        for p in self.particles:
            size = max(0, int(p['life'] / 5))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y), size, size))

        # Draw robot
        self._draw_robot()

    def _draw_robot(self):
        robot_rect = self.get_robot_rect()
        
        # Dynamic shape for game feel
        squash = min(5, self.robot_vel.y * 0.5) if self.robot_vel.y > 0 else 0
        stretch = max(-10, self.robot_vel.y * 0.7) if self.robot_vel.y < 0 else 0
        
        dynamic_rect = pygame.Rect(
            robot_rect.x - squash / 2,
            robot_rect.y + stretch,
            robot_rect.width + squash,
            robot_rect.height - stretch
        )
        
        # Glow effect
        glow_rect = dynamic_rect.inflate(8, 8)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_GLOW, glow_rect, 3, border_radius=5)
        
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, dynamic_rect, border_radius=3)
        
        # Eye
        eye_x = dynamic_rect.centerx + 5
        eye_y = dynamic_rect.centery - 5
        pygame.draw.circle(self.screen, self.COLOR_TEXT, (eye_x, eye_y), 3)

    def _render_ui(self):
        # Display distance
        dist_text = f"Distance: {int(self.score)}m / {self.GOAL_DISTANCE_METERS}m"
        self._draw_text(dist_text, (10, 10), self.font_ui, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Game Over message
        if self.game_over:
            if self.goal_reached:
                msg = "GOAL REACHED!"
            else:
                msg = "GAME OVER"
            
            self._draw_text(msg, 
                            (self.WIDTH // 2, self.HEIGHT // 2 - 20), 
                            self.font_game_over, 
                            self.COLOR_TEXT, 
                            self.COLOR_TEXT_SHADOW, 
                            center=True)
            
    def _draw_text(self, text, pos, font, color, shadow_color, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
            
        # Draw shadow
        self.screen.blit(shadow_surf, text_rect.move(2, 2))
        # Draw main text
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_meters": int(self.score),
        }

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup Pygame for human play
    pygame.display.set_caption("Robot Runner")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # Human controls
        keys = pygame.key.get_pressed()
        space_held = keys[pygame.K_SPACE]
        
        action = env.action_space.sample() # Start with a random action
        action[1] = 1 if space_held else 0 # Override with human input
        
        # For quitting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
        
        # Render to screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()