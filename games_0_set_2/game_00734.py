
# Generated: 2025-08-27T14:35:56.384077
# Source Brief: brief_00734.md
# Brief Index: 734

        
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
        "Controls: ←→ to run, ↑ to jump. Reach the green flag at the end of each stage."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced retro platformer. Guide a robot through procedurally generated levels, "
        "avoiding obstacles and trying to finish all three stages as quickly as possible."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GROUND_Y = 350
        
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
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 52)
            
        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (25, 30, 50)
        self.COLOR_GROUND = (60, 65, 90)
        self.COLOR_ROBOT = (60, 160, 255)
        self.COLOR_ROBOT_GLOW = (120, 200, 255)
        self.COLOR_OBSTACLE = (255, 80, 80)
        self.COLOR_OBSTACLE_GLOW = (255, 140, 140)
        self.COLOR_FLAG = (80, 255, 80)
        self.COLOR_TEXT = (240, 240, 240)
        
        # Physics constants
        self.GRAVITY = 0.6
        self.JUMP_STRENGTH = -13
        self.MOVE_SPEED = 6.0
        self.FRICTION = 0.85

        # Initialize state variables to be defined in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.lives = 0
        self.stage = 0
        self.game_timer = 0.0
        self.camera_x = 0.0
        
        self.robot_pos = [0.0, 0.0]
        self.robot_vel = [0.0, 0.0]
        self.robot_size = (30, 50)
        self.robot_on_ground = False
        self.robot_rect = pygame.Rect(0, 0, 0, 0)
        
        self.obstacles = []
        self.end_flag_rect = None
        self.particles = []
        
        # Initialize state
        self.reset()
        
        # Self-validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.lives = 3
        self.stage = 1
        self.game_timer = 0.0
        self.particles.clear()
        
        self._generate_stage()
        self._reset_player_position()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()

        self.steps += 1
        self.game_timer += 1 / 30.0
        
        # Unpack factorized action
        movement = action[0]
        
        self._handle_input(movement)
        events = self._update_physics()
        
        # --- Reward Calculation and State Updates ---
        if events["moving_right"]:
            reward += 0.1
        if not self.robot_on_ground:
            reward -= 0.01

        if events["hit_obstacle"]:
            reward -= 1.0
            self.lives -= 1
            # sfx: player_hit_sfx
            self._create_particles(self.robot_rect.center, self.COLOR_OBSTACLE, 20, 4)
            self._reset_player_position()

        if events["fell_off"]:
            reward -= 10.0
            self.lives -= 1
            # sfx: player_fall_sfx
            self._reset_player_position()

        if events["finished_stage"]:
            reward += 10.0
            self.score += 1000 # Stage clear bonus
            self.stage += 1
            if self.stage > 3:
                self.win = True
                self.game_over = True
                reward += 50.0 # Final win bonus
                # sfx: game_win_sfx
            else:
                # sfx: stage_clear_sfx
                self._generate_stage()
                self._reset_player_position()

        self.score += reward
        
        # --- Termination Check ---
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1 and self.robot_on_ground: # Jump
            self.robot_vel[1] = self.JUMP_STRENGTH
            self.robot_on_ground = False
            # sfx: jump_sfx
            self._create_particles(self.robot_rect.midbottom, self.COLOR_ROBOT_GLOW, 10, 2)
        elif movement == 3: # Left
            self.robot_vel[0] -= self.MOVE_SPEED * 0.2 # Make left/right acceleration based
        elif movement == 4: # Right
            self.robot_vel[0] += self.MOVE_SPEED * 0.2

    def _update_physics(self):
        events = {"hit_obstacle": False, "fell_off": False, "finished_stage": False, "moving_right": False}

        # Apply friction and cap velocity
        self.robot_vel[0] *= self.FRICTION
        self.robot_vel[0] = max(-self.MOVE_SPEED, min(self.MOVE_SPEED, self.robot_vel[0]))
        
        # Horizontal Movement
        if abs(self.robot_vel[0]) > 0.1:
            self.robot_pos[0] += self.robot_vel[0]
            if self.robot_vel[0] > 0:
                events["moving_right"] = True
        
        self.robot_pos[0] = max(self.robot_pos[0], 0)

        # Vertical Movement (Gravity)
        if not self.robot_on_ground:
            self.robot_vel[1] += self.GRAVITY
        self.robot_pos[1] += self.robot_vel[1]
        
        # Update robot rect for collisions
        self.robot_rect = pygame.Rect(
            int(self.robot_pos[0] - self.robot_size[0] / 2),
            int(self.robot_pos[1] - self.robot_size[1] / 2),
            *self.robot_size
        )

        # Ground Collision
        was_on_ground = self.robot_on_ground
        if self.robot_rect.bottom >= self.GROUND_Y and self.robot_vel[1] >= 0:
            self.robot_rect.bottom = self.GROUND_Y
            self.robot_pos[1] = self.robot_rect.centery
            self.robot_vel[1] = 0
            self.robot_on_ground = True
            if not was_on_ground:
                # sfx: land_sfx
                self._create_particles(self.robot_rect.midbottom, self.COLOR_GROUND, 5, 1)
        else:
            self.robot_on_ground = False

        # Fall off check
        if self.robot_rect.top > self.SCREEN_HEIGHT:
            events["fell_off"] = True

        # Obstacle Collision
        for obs in self.obstacles:
            if self.robot_rect.colliderect(obs):
                events["hit_obstacle"] = True
                break
        
        # End Flag Collision
        if self.end_flag_rect and self.robot_rect.colliderect(self.end_flag_rect):
            events["finished_stage"] = True

        # Update camera
        self.camera_x = self.robot_pos[0] - self.SCREEN_WIDTH / 3.0
        
        self._update_particles()
        
        return events

    def _generate_stage(self):
        self.obstacles.clear()
        stage_length = 2000 + 1500 * self.stage
        obstacle_density_factor = 1.0 + (self.stage - 1) * 0.1
        
        current_x = 600
        while current_x < stage_length - 800:
            gap = self.np_random.uniform(250, 450) / obstacle_density_factor
            current_x += gap
            
            width = self.np_random.integers(30, 81)
            height = self.np_random.integers(40, 101)
            
            self.obstacles.append(pygame.Rect(
                int(current_x), self.GROUND_Y - height, width, height
            ))
            
        flag_pos = (stage_length - 100, self.GROUND_Y - 100)
        self.end_flag_rect = pygame.Rect(*flag_pos, 20, 100)

    def _reset_player_position(self):
        start_x = self.camera_x + 100 if self.camera_x > 0 else 100
        self.robot_pos = [start_x, self.GROUND_Y - 100.0]
        self.robot_vel = [0.0, 0.0]
        self.robot_on_ground = False

    def _check_termination(self):
        if self.lives <= 0:
            self.game_over = True
            # sfx: game_over_sfx
        if self.steps >= 2000:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw background grid
        for i in range(0, self.SCREEN_WIDTH, 50):
            x = i - int(self.camera_x * 0.1) % 50
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 50):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i))

        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.GROUND_Y))

        # Draw obstacles
        for obs in self.obstacles:
            screen_obs = obs.move(-int(self.camera_x), 0)
            if screen_obs.right > 0 and screen_obs.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_obs)
                glow_rect = screen_obs.inflate(4, 4)
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE_GLOW, glow_rect, 1, border_radius=2)

        # Draw end flag
        if self.end_flag_rect:
            screen_flag = self.end_flag_rect.move(-int(self.camera_x), 0)
            if screen_flag.right > 0 and screen_flag.left < self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_FLAG, screen_flag)
                pygame.draw.polygon(self.screen, self.COLOR_FLAG, [
                    (screen_flag.right, screen_flag.top),
                    (screen_flag.right + 40, screen_flag.top + 20),
                    (screen_flag.right, screen_flag.top + 40)
                ])

        # Draw particles
        for p in self.particles:
            p_pos = (int(p['pos'][0] - self.camera_x), int(p['pos'][1]))
            alpha = int(255 * (p['life'] / p['max_life']))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], int(p['size']), color)

        # Draw robot
        screen_robot_rect = self.robot_rect.move(-int(self.camera_x), 0)
        
        # Bobbing animation when running on ground
        bob = 0
        if self.robot_on_ground and abs(self.robot_vel[0]) > 0.5:
            bob = abs(math.sin(self.game_timer * 20)) * -4
        
        # Apply bob to a copy of the rect
        anim_rect = screen_robot_rect.move(0, bob)
        
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, anim_rect, border_radius=3)
        glow_rect = anim_rect.inflate(6, 6)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT_GLOW, glow_rect, 1, border_radius=5)

    def _render_ui(self):
        # Lives display
        lives_text = self.font_main.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 10))

        # Timer display
        minutes, seconds = divmod(int(self.game_timer), 60)
        timer_text = self.font_main.render(f"TIME: {minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Stage display
        stage_text = self.font_main.render(f"STAGE: {self.stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH // 2 - stage_text.get_width() // 2, 10))
        
        # Game Over / Win message
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_FLAG if self.win else self.COLOR_OBSTACLE
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            msg_render = self.font_large.render(message, True, color)
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_render, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "stage": self.stage,
        }

    def _create_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos), 'vel': vel, 'life': life, 'max_life': life,
                'color': color, 'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # particle gravity
            p['life'] -= 1
            p['size'] -= 0.1
        self.particles = [p for p in self.particles if p['life'] > 0 and p['size'] > 0]

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to actions
    key_map = {
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_SPACE: 1, # Space is action[1]
        pygame.K_LSHIFT: 1, # Shift is action[2]
    }
    
    # Pygame setup for human play
    pygame.display.set_caption("Robot Platformer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    # Game loop for human play
    running = True
    while running:
        # Action defaults to no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        # Poll for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset() # Reset on 'r' key
        
        # Check held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1

        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            # Wait a bit before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()