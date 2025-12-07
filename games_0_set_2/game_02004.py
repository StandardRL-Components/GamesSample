
# Generated: 2025-08-27T18:57:01.833988
# Source Brief: brief_02004.md
# Brief Index: 2004

        
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
        "Controls: Use ← and → to run, and ↑ or Space to jump. Avoid obstacles to reach the finish line."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, side-scrolling platformer. Guide a robot through a procedurally generated "
        "obstacle course, timing your jumps to survive and reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.WORLD_WIDTH = 2500
        self.FINISH_LINE_X = self.WORLD_WIDTH - 150
        self.GROUND_Y = 360
        self.MAX_STEPS = 5000

        # Player physics
        self.PLAYER_SPEED = 4.0
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -12.0

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (30, 30, 50)
        self.COLOR_PLAYER = (255, 215, 0)
        self.COLOR_FINISH = (255, 255, 255)
        self.OBSTACLE_COLORS = {
            "green": (50, 205, 50),
            "blue": (65, 105, 225),
            "red": (220, 20, 60),
        }
        self.OBSTACLE_REWARDS = {"green": 0.2, "blue": 0.5, "red": 1.0}

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)

        # --- State Variables ---
        self.player_pos = None
        self.player_vel = None
        self.player_size = None
        self.on_ground = None
        self.camera_x = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.obstacle_base_speed = None
        self.next_obstacle_spawn_x = None

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([100.0, self.GROUND_Y - 40.0])
        self.player_vel = np.array([0.0, 0.0])
        self.player_size = np.array([20, 40])
        self.on_ground = True
        self.camera_x = 0.0
        self.obstacles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.obstacle_base_speed = 2.0
        self.next_obstacle_spawn_x = 400

        # Spawn initial obstacles
        while self.next_obstacle_spawn_x < self.WORLD_WIDTH - 300:
            self._spawn_obstacle()
            
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # --- Update game logic based on action ---
        self._handle_input(action)
        self._update_player()
        self._update_world()

        # --- Calculate reward and check for termination ---
        reward, terminated = self._calculate_reward_and_check_termination()
        
        self.score += reward
        self.steps += 1
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel[0] = self.PLAYER_SPEED
        else:
            self.player_vel[0] = 0

        # Jumping
        if (movement == 1 or space_held) and self.on_ground:
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump_sound

    def _update_player(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY

        # Update position
        self.player_pos += self.player_vel

        # Ground collision
        if self.player_pos[1] + self.player_size[1] >= self.GROUND_Y:
            self.player_pos[1] = self.GROUND_Y - self.player_size[1]
            self.player_vel[1] = 0
            if not self.on_ground:
                # sfx: land_sound
                self._create_particles(self.player_pos + np.array([self.player_size[0]/2, self.player_size[1]]), self.COLOR_PLAYER, 5, 2.0)
            self.on_ground = True

        # World bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WORLD_WIDTH - self.player_size[0])
        
        # Update camera with smoothing
        target_camera_x = self.player_pos[0] - self.SCREEN_WIDTH / 3
        self.camera_x = self.camera_x * 0.9 + target_camera_x * 0.1
        self.camera_x = np.clip(self.camera_x, 0, self.WORLD_WIDTH - self.SCREEN_WIDTH)

    def _update_world(self):
        # Update obstacle difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_base_speed += 0.1
            
        # Update obstacles
        for obs in self.obstacles:
            obs['pos'][0] += obs['vel']
        
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _calculate_reward_and_check_termination(self):
        reward = 0.0
        terminated = False
        
        # Survival reward
        reward += 0.01

        player_rect = pygame.Rect(self.player_pos, self.player_size)

        # Check obstacle collision and passing reward
        obstacle_is_near = False
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'], obs['size'])
            
            # Collision check
            if player_rect.colliderect(obs_rect):
                reward = -10.0
                terminated = True
                self._create_particles(player_rect.center, self.OBSTACLE_COLORS[obs['type']], 30, 5.0)
                # sfx: explosion_sound
                break
            
            # Passing reward
            if not obs['passed'] and obs_rect.right < player_rect.left:
                reward += self.OBSTACLE_REWARDS[obs['type']]
                obs['passed'] = True

            # Check for nearby obstacles for safe action penalty
            if obs_rect.left > player_rect.right and obs_rect.left - player_rect.right < 250:
                obstacle_is_near = True

        if terminated: # Early exit on collision
            return reward, terminated
            
        # Safe action penalty (jumping when not necessary)
        if not self.on_ground and not obstacle_is_near:
            reward -= 0.02

        # Finish line reward
        if player_rect.centerx >= self.FINISH_LINE_X:
            reward += 100.0
            terminated = True
            self.win = True
            # sfx: victory_sound

        # Timeout termination
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        return reward, terminated

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Background Grid (Parallax) ---
        cam_int = int(self.camera_x)
        for i in range(0, self.SCREEN_WIDTH + 100, 50):
            x = i - (cam_int % 50)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT + 100, 50):
            y = i
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
            
        # --- Draw Ground ---
        ground_rect = pygame.Rect(0, self.GROUND_Y - cam_int, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 3)

        # --- Draw Finish Line ---
        finish_x_on_screen = self.FINISH_LINE_X - cam_int
        if 0 < finish_x_on_screen < self.SCREEN_WIDTH:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (finish_x_on_screen, 0), (finish_x_on_screen, self.GROUND_Y), 5)
            for i in range(0, self.GROUND_Y, 20):
                 pygame.draw.rect(self.screen, self.COLOR_FINISH, (finish_x_on_screen, i, 10, 10))

        # --- Draw Obstacles ---
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['pos'] - (self.camera_x, 0), obs['size'])
            if obs_rect.right > 0 and obs_rect.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.OBSTACLE_COLORS[obs['type']], obs_rect, border_radius=3)
                pygame.draw.rect(self.screen, tuple(min(255, c+40) for c in self.OBSTACLE_COLORS[obs['type']]), obs_rect.inflate(-6, -6), border_radius=3)

        # --- Draw Player ---
        if not (self.game_over and not self.win): # Don't draw player if they lost
            player_draw_pos = self.player_pos - (self.camera_x, 0)
            player_rect = pygame.Rect(player_draw_pos, self.player_size)
            
            # Simple animation
            if not self.on_ground: # Stretch when jumping
                player_rect.height += 5
                player_rect.y -= 5
                player_rect.width -= 4
                player_rect.x += 2
            else: # Bob when running
                bob = math.sin(self.steps * 0.5) * 2
                player_rect.y += bob
            
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
            # "Eye"
            eye_pos = (int(player_rect.centerx + 5), int(player_rect.y + 10))
            pygame.draw.circle(self.screen, self.COLOR_BG, eye_pos, 3)

        # --- Draw Particles ---
        for p in self.particles:
            pos = p['pos'] - (self.camera_x, 0)
            alpha = int(255 * (p['life'] / p['max_life']))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(pos[0]), int(pos[1]), int(p['size']), (*p['color'], alpha)
                )

    def _render_ui(self):
        # Score and Time
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_FINISH)
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left//60:02}:{time_left%60:02}", True, self.COLOR_FINISH)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        self.screen.blit(time_text, (10, 10))

        # Game Over / Win Message
        if self.game_over:
            if self.win:
                msg_text = self.font_msg.render("LEVEL COMPLETE!", True, self.COLOR_PLAYER)
            else:
                msg_text = self.font_msg.render("GAME OVER", True, self.OBSTACLE_COLORS['red'])
            
            msg_rect = msg_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_text, msg_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _spawn_obstacle(self):
        obs_type = self.np_random.choice(["green", "blue", "red"], p=[0.5, 0.3, 0.2])
        
        height = self.np_random.integers(25, 61)
        width = self.np_random.integers(20, 41)
        
        x_pos = self.next_obstacle_spawn_x
        y_pos = self.GROUND_Y - height
        
        speed_multiplier = {"green": 0.8, "blue": 1.0, "red": 1.2}
        velocity = -self.obstacle_base_speed * speed_multiplier[obs_type]

        self.obstacles.append({
            "pos": np.array([float(x_pos), float(y_pos)]),
            "size": np.array([width, height]),
            "vel": velocity,
            "type": obs_type,
            "passed": False
        })

        gap = self.np_random.integers(150, 401)
        self.next_obstacle_spawn_x += width + gap
        
    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(20, 40)
            self.particles.append({
                "pos": pos.copy().astype(float),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
                "size": self.np_random.integers(2, 6)
            })

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption(env.game_description)
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    total_reward = 0

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            # --- Action Mapping for Human ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        else:
             # Allow reset on key press after game over
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                obs, info = env.reset()
                terminated = False
                total_reward = 0


        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2)) # Transpose back for pygame display
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS

    pygame.quit()