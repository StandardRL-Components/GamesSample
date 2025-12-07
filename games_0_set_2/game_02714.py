
# Generated: 2025-08-27T21:14:21.281751
# Source Brief: brief_02714.md
# Brief Index: 2714

        
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
        "Controls: ↑↓ to move vertically. ←→ to nudge horizontally. "
        "Hold Space to accelerate and Shift to brake."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race your neon robot down a procedurally generated track, dodging "
        "obstacles to set the fastest time. Finish under 20 seconds to win, "
        "but crash 5 times and you're out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.TRACK_Y_TOP = 80
        self.TRACK_Y_BOTTOM = 320
        self.TRACK_HEIGHT = self.TRACK_Y_BOTTOM - self.TRACK_Y_TOP
        self.TRACK_LENGTH = 12000  # World units
        self.PLAYER_START_X = 200.0
        self.PLAYER_SCREEN_X = 160 # Fixed screen x-position for the player

        # Game rules
        self.WIN_TIME = 20.0
        self.MAX_COLLISIONS = 5
        self.MAX_STEPS = 5000
        self.FPS = 30

        # Colors (Neon/Retro-futuristic)
        self.COLOR_BG = (10, 10, 30)
        self.COLOR_TRACK = (200, 200, 255)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_OBSTACLE = (255, 0, 85)
        self.COLOR_OBSTACLE_GLOW = (150, 0, 50)
        self.COLOR_PARTICLE = (255, 204, 0)
        self.COLOR_FINISH_A = (0, 255, 0)
        self.COLOR_FINISH_B = (0, 170, 0)
        self.COLOR_CHECKPOINT = (0, 100, 255)
        self.COLOR_UI = (255, 255, 255)

        # Physics constants
        self.ACCELERATION = 0.4
        self.BRAKING_FORCE = 0.8
        self.FRICTION_H = 0.985
        self.CRUISE_SPEED = 8.0
        self.MAX_SPEED = 25.0
        self.MIN_SPEED = 4.0
        self.VERTICAL_SPEED = 6.0
        self.FRICTION_V = 0.80
        self.NUDGE_SPEED = 4.0
        self.NUDGE_DECAY = 0.90
        self.COLLISION_KNOCKBACK = -5.0

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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 72)

        # Etc...
        self.obstacles = []
        self.particles = []
        self.speed_lines = []
        self.checkpoints = []

        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.player_pos = [self.PLAYER_START_X, self.SCREEN_HEIGHT / 2.0]
        self.player_vel = [self.CRUISE_SPEED, 0.0]
        self.player_nudge_offset = 0.0

        self.camera_x = 0.0
        self.elapsed_time = 0.0
        self.collisions = 0
        
        self.obstacle_density = 0.1
        self.next_obstacle_spawn_x = self.SCREEN_WIDTH

        self.obstacles.clear()
        self.particles.clear()
        self.speed_lines.clear()
        self.checkpoints.clear()
        
        self._generate_checkpoints()
        self.next_checkpoint_idx = 0
        
        # Pre-populate the screen with obstacles
        while self.next_obstacle_spawn_x < self.camera_x + self.SCREEN_WIDTH:
            self._spawn_obstacle_column()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.steps += 1
        self.elapsed_time += 1.0 / self.FPS
        reward = 0.0

        # --- 1. Process Actions ---
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Vertical movement
        if movement == 1: # Up
            self.player_vel[1] -= self.VERTICAL_SPEED / self.FPS
        elif movement == 2: # Down
            self.player_vel[1] += self.VERTICAL_SPEED / self.FPS
            
        # Horizontal nudge
        if movement == 3: # Left
            self.player_nudge_offset -= self.NUDGE_SPEED
        elif movement == 4: # Right
            self.player_nudge_offset += self.NUDGE_SPEED

        # Speed control
        if space_held:
            self.player_vel[0] += self.ACCELERATION
        if shift_held:
            self.player_vel[0] -= self.BRAKING_FORCE

        # --- 2. Update Physics ---
        # Apply friction and decay
        self.player_vel[0] *= self.FRICTION_H
        self.player_vel[0] = max(self.MIN_SPEED, min(self.player_vel[0], self.MAX_SPEED))
        
        self.player_vel[1] *= self.FRICTION_V
        self.player_nudge_offset *= self.NUDGE_DECAY

        # Update position
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        
        # Clamp player to track boundaries
        self.player_pos[1] = np.clip(self.player_pos[1], self.TRACK_Y_TOP + 10, self.TRACK_Y_BOTTOM - 10)
        self.player_nudge_offset = np.clip(self.player_nudge_offset, -self.PLAYER_SCREEN_X / 2, self.PLAYER_SCREEN_X / 2)

        # --- 3. Update Game World ---
        self.camera_x = self.player_pos[0] - self.PLAYER_SCREEN_X
        
        # Update difficulty
        if self.steps > 0 and self.steps % 500 == 0:
            self.obstacle_density = min(0.4, self.obstacle_density + 0.01)

        # Procedurally generate obstacles
        if self.camera_x + self.SCREEN_WIDTH > self.next_obstacle_spawn_x:
            self._spawn_obstacle_column()
            
        # Update and prune dynamic elements
        self._update_particles()
        self._update_speed_lines()
        self.obstacles = [obs for obs in self.obstacles if obs['pos'][0] > self.camera_x - 50]

        # --- 4. Check for Events & Rewards ---
        step_reward = self._calculate_reward()
        
        # --- 5. Check Termination Conditions ---
        terminated = self._check_termination()
        
        # Add terminal rewards if game just ended
        if terminated and not self.game_over:
             if self.win_condition_met:
                step_reward += 100.0 # Win bonus
             elif self.collisions >= self.MAX_COLLISIONS:
                step_reward -= 100.0 # Loss penalty
             else: # Timeout
                step_reward -= 50.0

        if terminated:
            self.game_over = True
        
        self.score += step_reward
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            step_reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self):
        reward = 0.1 # Survival reward
        
        # Collision check
        player_rect = pygame.Rect(self.player_pos[0] - 8, self.player_pos[1] - 8, 16, 16)
        for obs in self.obstacles:
            if not obs.get('hit', False):
                dist = math.hypot(player_rect.centerx - obs['pos'][0], player_rect.centery - obs['pos'][1])
                if dist < (player_rect.width / 2 + obs['radius']):
                    self.collisions += 1
                    obs['hit'] = True
                    reward -= 10.0 # Collision penalty
                    self.player_vel[0] = max(self.MIN_SPEED, self.player_vel[0] + self.COLLISION_KNOCKBACK)
                    self._create_collision_particles(self.player_pos)
                    # sfx: explosion

        # Checkpoint check
        if self.next_checkpoint_idx < len(self.checkpoints):
            checkpoint_x = self.checkpoints[self.next_checkpoint_idx]
            if self.player_pos[0] > checkpoint_x:
                reward += 5.0
                self.next_checkpoint_idx += 1
                # sfx: checkpoint_pass
        return reward

    def _check_termination(self):
        # Win condition
        if self.player_pos[0] >= self.TRACK_LENGTH:
            if self.elapsed_time <= self.WIN_TIME:
                self.win_condition_met = True
            return True
        # Loss conditions
        if self.collisions >= self.MAX_COLLISIONS:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "collisions": self.collisions, "time": self.elapsed_time}

    def _render_game(self):
        # Draw track boundaries
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_TOP), (self.SCREEN_WIDTH, self.TRACK_Y_TOP), 2)
        pygame.draw.line(self.screen, self.COLOR_TRACK, (0, self.TRACK_Y_BOTTOM), (self.SCREEN_WIDTH, self.TRACK_Y_BOTTOM), 2)

        # Draw checkpoints
        for i, cp_x in enumerate(self.checkpoints):
            screen_x = int(cp_x - self.camera_x)
            if 0 <= screen_x <= self.SCREEN_WIDTH:
                color = self.COLOR_CHECKPOINT if i >= self.next_checkpoint_idx else self.COLOR_FINISH_A
                pygame.draw.line(self.screen, color, (screen_x, self.TRACK_Y_TOP), (screen_x, self.TRACK_Y_BOTTOM), 3)

        # Draw finish line
        finish_x = int(self.TRACK_LENGTH - self.camera_x)
        if 0 <= finish_x <= self.SCREEN_WIDTH:
            check_size = 20
            for y in range(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM, check_size):
                for x_offset in range(0, 2 * check_size, check_size):
                    color = self.COLOR_FINISH_A if ((y // check_size) % 2 == (x_offset // check_size) % 2) else self.COLOR_FINISH_B
                    pygame.draw.rect(self.screen, color, (finish_x + x_offset - check_size, y, check_size, check_size))

        # Draw obstacles
        for obs in self.obstacles:
            sx = int(obs['pos'][0] - self.camera_x)
            sy = int(obs['pos'][1])
            radius = int(obs['radius'])
            if -radius < sx < self.SCREEN_WIDTH + radius:
                color_glow = self.COLOR_OBSTACLE_GLOW if not obs.get('hit') else (80,80,80)
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, color_glow)
                color_main = self.COLOR_OBSTACLE if not obs.get('hit') else (120,120,120)
                pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, color_main)

        # Draw speed lines
        for line in self.speed_lines:
            start_pos = (int(line['start'][0] - self.camera_x), int(line['start'][1]))
            end_pos = (int(line['end'][0] - self.camera_x), int(line['end'][1]))
            pygame.draw.line(self.screen, line['color'], start_pos, end_pos, line['width'])
            
        # Draw particles
        for p in self.particles:
            sx = int(p['pos'][0] - self.camera_x)
            sy = int(p['pos'][1])
            pygame.draw.circle(self.screen, p['color'], (sx, sy), int(p['size']))

        # Draw player
        player_screen_pos_x = int(self.PLAYER_SCREEN_X + self.player_nudge_offset)
        player_screen_pos_y = int(self.player_pos[1])
        player_size = 8
        
        # Glow effect
        glow_size = int(player_size * (1.5 + self.player_vel[0] / self.MAX_SPEED))
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW + (80,), (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (player_screen_pos_x - glow_size, player_screen_pos_y - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player body
        player_rect = pygame.Rect(player_screen_pos_x - player_size, player_screen_pos_y - player_size, player_size*2, player_size*2)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        # Time display
        time_text = f"Time: {self.elapsed_time:.2f}s"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI)
        self.screen.blit(time_surf, (10, 10))

        # Collision display
        collision_text = f"Collisions: {self.collisions}/{self.MAX_COLLISIONS}"
        collision_surf = self.font_ui.render(collision_text, True, self.COLOR_OBSTACLE if self.collisions > 0 else self.COLOR_UI)
        self.screen.blit(collision_surf, (self.SCREEN_WIDTH - collision_surf.get_width() - 10, 10))

        # Game Over / Win Message
        if self.game_over:
            if self.win_condition_met:
                msg_text = "YOU WIN!"
                msg_color = self.COLOR_FINISH_A
            else:
                msg_text = "GAME OVER"
                msg_color = self.COLOR_OBSTACLE
            
            msg_surf = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _generate_checkpoints(self):
        num_checkpoints = int(self.TRACK_LENGTH / 3000)
        for i in range(1, num_checkpoints + 1):
            self.checkpoints.append(i * (self.TRACK_LENGTH / (num_checkpoints + 1)))

    def _spawn_obstacle_column(self):
        spacing = self.np_random.uniform(250, 400)
        self.next_obstacle_spawn_x += spacing
        
        num_to_spawn = self.np_random.integers(1, 4)
        available_y_slots = list(range(self.TRACK_Y_TOP + 20, self.TRACK_Y_BOTTOM - 20, 40))
        
        for _ in range(num_to_spawn):
            if self.np_random.random() < self.obstacle_density and available_y_slots:
                y_idx = self.np_random.integers(0, len(available_y_slots))
                y_pos = available_y_slots.pop(y_idx)
                
                self.obstacles.append({
                    'pos': [self.next_obstacle_spawn_x, y_pos],
                    'radius': self.np_random.uniform(8, 15)
                })

    def _create_collision_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 8)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'size': self.np_random.uniform(2, 5),
                'life': self.np_random.uniform(15, 30) # in frames
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _update_speed_lines(self):
        # Generate new speed lines
        if self.player_vel[0] > self.CRUISE_SPEED + 2:
            num_lines = int((self.player_vel[0] - self.CRUISE_SPEED) / 4)
            for _ in range(num_lines):
                y = self.np_random.uniform(self.TRACK_Y_TOP, self.TRACK_Y_BOTTOM)
                length = self.player_vel[0] * 1.5
                self.speed_lines.append({
                    'start': [self.player_pos[0] - length, y],
                    'end': [self.player_pos[0], y],
                    'color': (100,100,150, 100),
                    'width': self.np_random.integers(1,3)
                })
        
        # Prune old speed lines
        self.speed_lines = [line for line in self.speed_lines if line['end'][0] > self.camera_x - 50]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
    env = GameEnv()
    env.validate_implementation()

    import sys
    
    pygame.display.set_caption("Arcade Racer")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Main game loop for manual play
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated:
            # Show final screen for a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        keys = pygame.key.get_pressed()
        
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    pygame.quit()
    sys.exit()