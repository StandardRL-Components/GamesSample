
# Generated: 2025-08-27T20:10:17.354123
# Source Brief: brief_02370.md
# Brief Index: 2370

        
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
        "Controls: ↑/←/→ for jumps, Space for boost jump, Shift to slide."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a robot running and jumping across a procedurally generated landscape to reach the finish line as fast as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1800  # 60 seconds * 30 FPS
        self.FINISH_DISTANCE = 8000  # Total pixels to travel

        # Colors
        self.COLOR_SKY = (135, 206, 235)
        self.COLOR_GROUND = (139, 69, 19)
        self.COLOR_ROBOT = (0, 128, 255)
        self.COLOR_ROBOT_GLOW = (0, 128, 255, 100)
        self.COLOR_OBSTACLE = (220, 20, 60)
        self.COLOR_FINISH = (0, 255, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # Physics & Gameplay
        self.GRAVITY = 1.2
        self.JUMP_SMALL = -15
        self.JUMP_MEDIUM = -18
        self.JUMP_HIGH = -21
        self.JUMP_BOOST = -25
        self.FAST_FALL_ACCEL = 3
        self.GROUND_Y = self.HEIGHT - 50
        self.INITIAL_SCROLL_SPEED = 6.0

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.distance_traveled = None
        self.scroll_speed = None
        self.obstacles = None
        self.particles = None
        self.cleared_obstacles_for_reward = None
        self.robot_rect = None
        self.robot_vy = None
        self.on_ground = None
        self.is_sliding = None
        self.slide_timer = None
        self._next_obstacle_x = None

        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.distance_traveled = 0
        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.obstacles = []
        self.particles = []
        self.cleared_obstacles_for_reward = set()

        # Initialize robot state
        self.robot_rect = pygame.Rect(100, self.GROUND_Y - 50, 30, 50)
        self.robot_vy = 0
        self.on_ground = True
        self.is_sliding = False
        self.slide_timer = 0
        
        # Procedural generation state
        self._next_obstacle_x = self.WIDTH * 1.5
        for _ in range(5):
            self._generate_obstacle(easy_mode=True)
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # 1. Handle Input & Update Robot State
            self._handle_input(action)
            self._update_robot_state()

            # 2. Update World State (Scrolling, Obstacle Generation)
            self._update_world_state()
            
            # 3. Collision Detection
            collision, finish = self._check_collisions()
            if collision:
                self.game_over = True
                reward = -10  # Penalty for collision
                # sfx: robot_crash.wav
            if finish:
                self.game_over = True
                time_bonus = max(0, self.MAX_STEPS - self.steps) / self.MAX_STEPS
                reward = 100 + (50 * time_bonus)  # Reward for finishing + time bonus
                # sfx: victory_fanfare.wav

            # 4. Calculate Step Reward
            reward += 0.01  # Small reward for surviving a step

            # Reward for clearing obstacles
            for obs in self.obstacles:
                obs_id = id(obs)
                if self.robot_rect.left > obs.right and obs_id not in self.cleared_obstacles_for_reward:
                    reward += 1
                    self.cleared_obstacles_for_reward.add(obs_id)
                    # sfx: clear_obstacle_chime.wav

        self.score += reward
        self.steps += 1
        
        # 5. Check Termination Conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
            reward -= 5  # Penalty for running out of time

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # Action Priority: Slide > Boost Jump > Regular Jump > Fast Fall
        if shift_pressed and self.on_ground and not self.is_sliding:
            self.is_sliding = True
            self.slide_timer = 15  # Slide duration in frames
            # sfx: slide_whoosh.wav
            return

        if self.is_sliding and not shift_pressed:
            self.is_sliding = False

        if self.on_ground and not self.is_sliding:
            jump_initiated = False
            if space_pressed:
                self.robot_vy = self.JUMP_BOOST
                jump_initiated = True
                # sfx: boost_jump.wav
            elif movement == 1: # Up
                self.robot_vy = self.JUMP_SMALL
                jump_initiated = True
                # sfx: jump_small.wav
            elif movement == 3: # Left
                self.robot_vy = self.JUMP_MEDIUM
                jump_initiated = True
                # sfx: jump_medium.wav
            elif movement == 4: # Right
                self.robot_vy = self.JUMP_HIGH
                jump_initiated = True
                # sfx: jump_high.wav
            
            if jump_initiated:
                self.on_ground = False
                self._create_particles(self.robot_rect.midbottom, 5, self.COLOR_GROUND)
        
        elif not self.on_ground and movement == 2: # Down (Fast Fall)
            self.robot_vy = min(self.robot_vy + self.FAST_FALL_ACCEL, 20)
            # sfx: fast_fall.wav

    def _update_robot_state(self):
        # Handle sliding timer
        if self.is_sliding:
            self.slide_timer -= 1
            if self.slide_timer <= 0:
                self.is_sliding = False
        
        # Apply gravity
        if not self.on_ground:
            self.robot_vy += self.GRAVITY
        
        # Update vertical position
        self.robot_rect.y += int(self.robot_vy)
        
        # Ground collision
        if self.robot_rect.bottom >= self.GROUND_Y:
            if not self.on_ground:  # Just landed
                self._create_particles(self.robot_rect.midbottom, 10, self.COLOR_GROUND)
                # sfx: landing_thud.wav
            self.robot_rect.bottom = self.GROUND_Y
            self.robot_vy = 0
            self.on_ground = True

    def _update_world_state(self):
        # Increase difficulty over time
        if self.steps > 0 and self.steps % 200 == 0:
            self.scroll_speed += 0.5
            self.scroll_speed = min(self.scroll_speed, 20)  # Cap speed

        self.distance_traveled += self.scroll_speed
        
        # Scroll obstacles
        for obs in self.obstacles:
            obs.x -= int(self.scroll_speed)
        
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs.right > 0]
        
        # Generate new obstacles
        if not self.obstacles or self.obstacles[-1].centerx < self.WIDTH:
            self._generate_obstacle()
            
        # Update particles
        for p in self.particles:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 1  # Decrement lifetime
        self.particles = [p for p in self.particles if p[2] > 0]

    def _generate_obstacle(self, easy_mode=False):
        difficulty_tier = self.steps // 200
        
        if easy_mode or self.steps < 50:
            gap = 250
            min_h, max_h = 20, 40
            min_w, max_w = 30, 50
        else:
            gap = random.randint(max(100, 200 - difficulty_tier * 10), 350 - difficulty_tier * 5)
            min_h = 20 + difficulty_tier * 5
            max_h = 60 + difficulty_tier * 10
            min_w = 30 + difficulty_tier * 2
            max_w = 70 + difficulty_tier * 5

        w = random.randint(min_w, max_w)
        h = random.randint(min_h, max_h)
        
        # Ensure obstacles are always jumpable with a boost jump
        max_jump_clearance = self.GROUND_Y - ((-self.JUMP_BOOST)**2 / (2 * self.GRAVITY))
        h = min(h, self.GROUND_Y - max_jump_clearance - 30) # 30px buffer
        h = max(h, 10)
        
        y = self.GROUND_Y - h
        x = self._next_obstacle_x
        
        self.obstacles.append(pygame.Rect(int(x), int(y), int(w), int(h)))
        self._next_obstacle_x += w + gap

    def _check_collisions(self):
        # Use a smaller hitbox when sliding
        hitbox = self.robot_rect.copy()
        if self.is_sliding:
            hitbox.height = 25
            hitbox.bottom = self.robot_rect.bottom
        
        # Obstacle collision
        if hitbox.collidelist(self.obstacles) != -1:
            return True, False
        
        # Finish line check
        finish_line_x_on_screen = self.FINISH_DISTANCE - self.distance_traveled
        if hitbox.right > finish_line_x_on_screen:
            return False, True
            
        return False, False
        
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
        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Finish Line
        finish_x = self.FINISH_DISTANCE - self.distance_traveled
        if finish_x < self.WIDTH + 20:
             pygame.draw.line(self.screen, self.COLOR_FINISH, (int(finish_x), 0), (int(finish_x), self.GROUND_Y), 5)

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)
            
        # Particles
        for p in self.particles:
            pos, _, lifetime = p
            alpha = max(0, min(255, lifetime * 20))
            s = pygame.Surface((lifetime, lifetime), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_GROUND, alpha), (lifetime//2, lifetime//2), lifetime//2)
            self.screen.blit(s, (int(pos[0]-lifetime//2), int(pos[1]-lifetime//2)))

        # Robot
        self._draw_robot()

    def _draw_robot(self):
        # Glow effect
        glow_rect = self.robot_rect.inflate(10, 10)
        s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_ROBOT_GLOW, s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_rect.topleft)

        # Body
        body_rect = self.robot_rect.copy()
        if self.is_sliding:
            body_rect.height = 25
            body_rect.bottom = self.robot_rect.bottom
            pygame.draw.rect(self.screen, self.COLOR_ROBOT, body_rect, border_radius=12)
        else:
            bob = math.sin(self.steps * 0.5) * 2 if self.on_ground else 0
            body_rect.y += bob
            pygame.draw.rect(self.screen, self.COLOR_ROBOT, body_rect, border_radius=5)
            
            # Eye
            eye_x = body_rect.centerx + 5
            eye_y = body_rect.centery - 10
            if not self.on_ground: # Wide eye when jumping
                 pygame.draw.rect(self.screen, (255, 255, 255), (eye_x-3, eye_y-2, 8, 4), border_radius=2)
            else: # Normal eye
                 pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), 3)

    def _render_ui(self):
        # Timer
        remaining_time = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"TIME: {remaining_time:.1f}"
        self._draw_text(time_text, self.font_small, (10, 10))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, self.font_small, (self.WIDTH - 150, 10))
        
        # Progress bar
        progress = min(1.0, self.distance_traveled / self.FINISH_DISTANCE)
        bar_width = self.WIDTH - 40
        bar_y = self.HEIGHT - 25
        bar_height = 10
        pygame.draw.rect(self.screen, (50, 50, 50), (20, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, (20, bar_y, bar_width * progress, bar_height), border_radius=5)

    def _draw_text(self, text, font, pos):
        text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, pos)
        
    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = random.uniform(math.pi, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.randint(10, 20)
            self.particles.append([[pos[0], pos[1]], vel, lifetime])
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_traveled": self.distance_traveled,
            "finish_distance": self.FINISH_DISTANCE
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
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