
# Generated: 2025-08-28T04:28:33.210802
# Source Brief: brief_02335.md
# Brief Index: 2335

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = "Controls: Use arrow keys to move the robot."

    # Short, user-facing description of the game
    game_description = (
        "A top-down arcade game where a robot collects parts while dodging deadly lasers to escape a factory."
    )

    # Frames auto-advance at a consistent rate
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500 # Increased to allow more time for completion
        self.VICTORY_PARTS = 15
        self.MAX_LIVES = 3

        # Colors
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_PART = (255, 200, 0)
        self.COLOR_PART_GLOW = (255, 200, 0, 80)
        self.COLOR_LASER = (255, 20, 20)
        self.COLOR_LASER_GLOW = (255, 20, 20, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)
        
        # Player settings
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 6
        self.INVINCIBILITY_FRAMES = 90 # 3 seconds at 30fps

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 50, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.lives = 0
        self.parts_collected = 0
        self.parts = None
        self.lasers = None
        self.particles = None
        self.invincibility_timer = 0
        self.laser_spawn_timer = 0
        self.laser_speed = 0
        self.steps_since_part = 0
        self.last_dist_to_part = 0
        self.screen_flash = 0

        self.reset()
        
        # self.validate_implementation() # Uncomment for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.lives = self.MAX_LIVES
        self.parts_collected = 0
        self.parts = []
        self.lasers = []
        self.particles = []
        self.invincibility_timer = 0
        self.laser_spawn_timer = 0
        self.laser_speed = 3.0
        self.steps_since_part = 0
        self.screen_flash = 0
        
        self._spawn_initial_parts(5)
        self.last_dist_to_part = self._get_dist_to_nearest_part()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        
        if not self.game_over:
            # --- Update State ---
            self.steps += 1
            self.steps_since_part += 1
            
            if self.invincibility_timer > 0:
                self.invincibility_timer -= 1
            if self.screen_flash > 0:
                self.screen_flash -= 1
            
            # --- Handle Input and Player Movement ---
            prev_pos = self.player_pos.copy()
            self._handle_movement(movement)
            
            # --- Update Game Objects ---
            self._update_lasers()
            self._update_particles()
            
            # --- Handle Collisions and Game Logic ---
            part_collected_this_step, hit_this_step, near_miss_this_step = self._handle_collisions()
            
            # --- Spawn New Objects ---
            self._spawn_lasers()
            self._handle_part_spawning(part_collected_this_step)
            
            # --- Calculate Reward ---
            reward = self._calculate_reward(prev_pos, part_collected_this_step, hit_this_step, near_miss_this_step)
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            terminal_reward = 100 if self.parts_collected >= self.VICTORY_PARTS else -100
            reward += terminal_reward
            self.score += terminal_reward
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info(),
        )

    def _calculate_reward(self, prev_pos, part_collected, hit, near_miss):
        reward = 0
        
        # 1. Survival reward
        reward += 0.01 # Smaller survival reward to not overpower other signals
        
        # 2. Part collection reward
        if part_collected:
            reward += 15.0
            
        # 3. Hit penalty
        if hit:
            reward -= 25.0
            
        # 4. Near miss reward
        if near_miss:
            reward += 0.5
            
        # 5. Movement reward (towards/away from part)
        current_dist_to_part = self._get_dist_to_nearest_part()
        if self.last_dist_to_part is not None and current_dist_to_part is not None:
            # Reward is proportional to distance closed. Max reward/penalty is ~0.2
            dist_delta = self.last_dist_to_part - current_dist_to_part
            reward += dist_delta * 0.025
        self.last_dist_to_part = current_dist_to_part
        
        self.score += reward
        return reward

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Clamp position to stay within screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

    def _update_lasers(self):
        for laser in self.lasers:
            laser['pos'] += laser['vel'] * self.laser_speed
        self.lasers = [l for l in self.lasers if -50 < l['pos'][0] < self.WIDTH + 50 and -50 < l['pos'][1] < self.HEIGHT + 50]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos - self.PLAYER_SIZE / 2, (self.PLAYER_SIZE, self.PLAYER_SIZE))
        part_collected = False
        hit = False
        near_miss = False

        # Part collisions
        for part in self.parts[:]:
            part_rect = pygame.Rect(part - 10, (20, 20))
            if player_rect.colliderect(part_rect):
                self.parts.remove(part)
                self.parts_collected += 1
                self.steps_since_part = 0
                part_collected = True
                self._create_particles(self.player_pos, self.COLOR_PART, 30)
                # Update difficulty
                if self.parts_collected % 3 == 0:
                    self.laser_speed += 0.25
                break # only collect one part per frame

        # Laser collisions
        if self.invincibility_timer == 0:
            near_miss_rect = player_rect.inflate(40, 40)
            for laser in self.lasers:
                laser_line = (laser['pos'], laser['pos'] + laser['dir'] * laser['len'])
                if player_rect.clipline(laser_line):
                    self.lives -= 1
                    hit = True
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    self.screen_flash = 10
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 20)
                    break
                elif near_miss_rect.clipline(laser_line):
                    near_miss = True
        
        return part_collected, hit, near_miss

    def _handle_part_spawning(self, part_collected_this_step):
        if part_collected_this_step and len(self.parts) < 5:
            self._spawn_part()
        if self.steps_since_part > 300: # Anti-softlock (10 seconds)
            self.parts.clear()
            self._spawn_initial_parts(5)
            self.steps_since_part = 0

    def _spawn_initial_parts(self, num_parts):
        for _ in range(num_parts):
            self._spawn_part()

    def _spawn_part(self):
        while True:
            pos = np.array([
                self.np_random.integers(30, self.WIDTH - 30),
                self.np_random.integers(30, self.HEIGHT - 30)
            ], dtype=np.float32)
            # Ensure it's not too close to the player or other parts
            if np.linalg.norm(pos - self.player_pos) > 100 and all(np.linalg.norm(pos - p) > 50 for p in self.parts):
                self.parts.append(pos)
                break
    
    def _spawn_lasers(self):
        self.laser_spawn_timer -= 1
        if self.laser_spawn_timer <= 0:
            difficulty = self.parts_collected // 3
            pattern = self.np_random.integers(0, min(difficulty + 1, 5))
            
            if pattern == 0: # Horizontal
                y = self.np_random.uniform(0.1, 0.9) * self.HEIGHT
                if self.np_random.random() < 0.5:
                    pos, vel, direction = np.array([-20, y]), np.array([1, 0]), np.array([1, 0])
                else:
                    pos, vel, direction = np.array([self.WIDTH+20, y]), np.array([-1, 0]), np.array([-1, 0])
                self.lasers.append({'pos': pos, 'vel': vel, 'dir': direction, 'len': 40})
                self.laser_spawn_timer = self.np_random.integers(30, 50)

            elif pattern == 1: # Vertical
                x = self.np_random.uniform(0.1, 0.9) * self.WIDTH
                if self.np_random.random() < 0.5:
                    pos, vel, direction = np.array([x, -20]), np.array([0, 1]), np.array([0, 1])
                else:
                    pos, vel, direction = np.array([x, self.HEIGHT+20]), np.array([0, -1]), np.array([0, -1])
                self.lasers.append({'pos': pos, 'vel': vel, 'dir': direction, 'len': 40})
                self.laser_spawn_timer = self.np_random.integers(30, 50)
            
            elif pattern == 2: # Sweeping
                side = self.np_random.integers(0, 4)
                if side == 0: # Left
                    start_pos = np.array([-20, self.np_random.uniform(0, self.HEIGHT)])
                    vel = np.array([1, self.np_random.uniform(-0.3, 0.3)])
                elif side == 1: # Right
                    start_pos = np.array([self.WIDTH+20, self.np_random.uniform(0, self.HEIGHT)])
                    vel = np.array([-1, self.np_random.uniform(-0.3, 0.3)])
                elif side == 2: # Top
                    start_pos = np.array([self.np_random.uniform(0, self.WIDTH), -20])
                    vel = np.array([self.np_random.uniform(-0.3, 0.3), 1])
                else: # Bottom
                    start_pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT+20])
                    vel = np.array([self.np_random.uniform(-0.3, 0.3), -1])
                vel /= np.linalg.norm(vel)
                self.lasers.append({'pos': start_pos, 'vel': vel, 'dir': vel, 'len': 40})
                self.laser_spawn_timer = self.np_random.integers(40, 60)

            elif pattern == 3: # Diagonal Burst
                for _ in range(self.np_random.integers(2, 4)):
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    vel = np.array([math.cos(angle), math.sin(angle)])
                    pos = self.player_pos - vel * 300 # spawn off-screen pointing at player
                    self.lasers.append({'pos': pos, 'vel': vel, 'dir': vel, 'len': 40})
                self.laser_spawn_timer = self.np_random.integers(60, 90)

            elif pattern == 4: # Laser Walls
                if self.np_random.random() < 0.5: # Vertical Wall
                    x = self.np_random.uniform(0.2, 0.8) * self.WIDTH
                    vel_dir = 1 if self.player_pos[0] < x else -1
                    for i in range(8):
                        pos = np.array([x + vel_dir * 100, i * self.HEIGHT/7 - 20])
                        vel = np.array([-vel_dir, 0])
                        self.lasers.append({'pos': pos, 'vel': vel, 'dir': np.array([0, 1]), 'len': self.HEIGHT/7})
                else: # Horizontal Wall
                    y = self.np_random.uniform(0.2, 0.8) * self.HEIGHT
                    vel_dir = 1 if self.player_pos[1] < y else -1
                    for i in range(12):
                        pos = np.array([i * self.WIDTH/11 - 20, y + vel_dir * 100])
                        vel = np.array([0, -vel_dir])
                        self.lasers.append({'pos': pos, 'vel': vel, 'dir': np.array([1, 0]), 'len': self.WIDTH/11})
                self.laser_spawn_timer = self.np_random.integers(90, 120)

    def _create_particles(self, position, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _get_dist_to_nearest_part(self):
        if not self.parts:
            return None
        distances = [np.linalg.norm(self.player_pos - p) for p in self.parts]
        return min(distances)

    def _check_termination(self):
        return (
            self.lives <= 0
            or self.parts_collected >= self.VICTORY_PARTS
            or self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "parts_collected": self.parts_collected,
        }

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        if self.screen_flash > 0:
            flash_alpha = 150 * (self.screen_flash / 10)
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((255, 100, 100, flash_alpha))
            self.screen.blit(flash_surface, (0, 0))
            
        self._render_game()
        self._render_ui()

    def _render_game(self):
        # Draw grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i), 1)

        # Draw parts
        for part_pos in self.parts:
            x, y = int(part_pos[0]), int(part_pos[1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 12, self.COLOR_PART_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 8, self.COLOR_PART)
            pygame.gfxdraw.aacircle(self.screen, x, y, 8, self.COLOR_PART)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (*p['color'], alpha)
            size = int(max(1, p['life'] / 6))
            pygame.draw.circle(self.screen, color, p['pos'].astype(int), size)

        # Draw lasers
        for laser in self.lasers:
            start = laser['pos'].astype(int)
            end = (laser['pos'] + laser['dir'] * laser['len']).astype(int)
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, start, end, 7)
            pygame.draw.line(self.screen, self.COLOR_LASER, start, end, 3)

        # Draw player
        is_invincible = self.invincibility_timer > 0
        if not (is_invincible and (self.steps // 3) % 2):
            px, py = int(self.player_pos[0]), int(self.player_pos[1])
            size = self.PLAYER_SIZE
            glow_size = int(size * 1.8)
            
            # Glow effect
            glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, glow_size, glow_size), border_radius=5)
            self.screen.blit(glow_surf, (px - glow_size // 2, py - glow_size // 2))

            # Main body
            player_rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, player_rect, width=1, border_radius=3)

    def _render_text(self, text, font, x, y, color, shadow_color):
        text_surf = font.render(text, True, shadow_color)
        self.screen.blit(text_surf, (x + 2, y + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))

    def _render_ui(self):
        # Render lives (hearts)
        for i in range(self.lives):
            x, y = 25 + i * 35, 30
            p1 = (x, y - 5)
            p2 = (x + 10, y - 15)
            p3 = (x + 20, y - 5)
            p4 = (x, y + 15)
            p5 = (x - 20, y - 5)
            p6 = (x - 10, y - 15)
            pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3,p4,p5,p6], self.COLOR_LASER)
            pygame.gfxdraw.aapolygon(self.screen, [p1,p2,p3,p4,p5,p6], self.COLOR_LASER)

        # Render parts collected
        parts_text = f"PARTS: {self.parts_collected}/{self.VICTORY_PARTS}"
        self._render_text(parts_text, self.font_ui, self.WIDTH - 220, 20, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        # Render score
        score_text = f"SCORE: {int(self.score)}"
        self._render_text(score_text, self.font_ui, self.WIDTH/2 - self.font_ui.size(score_text)[0]/2, self.HEIGHT - 40, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Render game over/win message
        if self.game_over:
            if self.parts_collected >= self.VICTORY_PARTS:
                msg = "YOU WIN!"
                color = self.COLOR_PART
            else:
                msg = "GAME OVER"
                color = self.COLOR_LASER
            
            text_w, text_h = self.font_big.size(msg)
            self._render_text(msg, self.font_big, self.WIDTH/2 - text_w/2, self.HEIGHT/2 - text_h/2, color, self.COLOR_TEXT_SHADOW)
            
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Playable Demo ---
    # This part is for demonstration and debugging.
    # It is not part of the Gymnasium environment itself.
    
    # Overwrite the screen to be a display surface for human play
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Robot Factory Escape")

    obs, info = env.reset()
    done = False
    
    # Game loop for human control
    while not done:
        # Action defaults to NO-OP
        movement = 0 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        # Space and Shift are not used in this game, so they are 0
        action = [movement, 0, 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the game to the display window
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Parts: {info['parts_collected']}")
            pygame.time.wait(3000) # Wait 3 seconds before resetting
            obs, info = env.reset()

        env.clock.tick(env.FPS)
        
    env.close()