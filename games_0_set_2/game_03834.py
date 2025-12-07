
# Generated: 2025-08-28T00:34:45.532485
# Source Brief: brief_03834.md
# Brief Index: 3834

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Collect coins and reach the green flag!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated platformer. Navigate tricky jumps, "
        "collect coins, and dodge falling hazards to reach the goal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 120, 120, 50)
    COLOR_COIN = (255, 215, 0)
    COLOR_PLATFORM = (100, 110, 130)
    COLOR_OBSTACLE = (180, 80, 255)
    COLOR_FLAG = (80, 220, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_PARTICLE_JUMP = (150, 150, 150)
    COLOR_PARTICLE_COIN = (255, 225, 100)
    COLOR_PARTICLE_DEATH = (255, 80, 80)

    # Physics
    GRAVITY = 0.4
    PLAYER_JUMP_STRENGTH = -9.5
    PLAYER_ACCEL = 0.9
    PLAYER_FRICTION = -0.15
    PLAYER_MAX_SPEED_X = 6.0
    PLAYER_MAX_SPEED_Y = 12.0

    # Game settings
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    INITIAL_LIVES = 3
    LEVEL_WIDTH_PIXELS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.level_num = 1
        self.terminated_by_win = False
        
        # This will be properly initialized in reset()
        self.player = {}
        self.platforms = []
        self.coins = []
        self.obstacles = []
        self.particles = []
        self.flag = None
        self.camera_offset = np.array([0.0, 0.0])
        self.rng = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.terminated_by_win:
            self.level_num += 1
        self.terminated_by_win = False

        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.lives = self.INITIAL_LIVES
        self.game_over = False
        
        self._generate_level()

        self.player = {
            "pos": np.array([150.0, 200.0]),
            "vel": np.array([0.0, 0.0]),
            "rect": pygame.Rect(0, 0, 20, 30),
            "is_grounded": False,
            "jump_cooldown": 0,
        }
        
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Update Game Logic ---
        self.steps += 1
        
        old_player_pos_x = self.player["pos"][0]

        # Handle input
        self._handle_input(movement, space_held)
        
        # Update entities
        self._update_player()
        self._update_obstacles()
        self._update_particles()
        
        # Handle collisions and collect event rewards
        reward += self._handle_collisions()

        # Update camera
        self._update_camera()

        # Calculate continuous rewards
        dx = self.player["pos"][0] - old_player_pos_x
        if dx > 0:
            reward += dx * 0.01  # Scaled down from brief for better balance
        else:
            reward += dx * 0.001 # Smaller penalty for moving left

        # Check for termination
        terminated = self._check_termination()
        if terminated and self.player["rect"].colliderect(self.flag):
            reward += 100
            self.terminated_by_win = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3:  # Left
            self.player["vel"][0] -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player["vel"][0] += self.PLAYER_ACCEL
        
        # Jumping
        is_trying_to_jump = (movement == 1 or space_held)
        if is_trying_to_jump and self.player["is_grounded"] and self.player["jump_cooldown"] == 0:
            self.player["vel"][1] = self.PLAYER_JUMP_STRENGTH
            self.player["is_grounded"] = False
            self.player["jump_cooldown"] = 5 # Small cooldown to prevent multi-jumps
            # SFX: Jump
            self._create_particles(self.player["rect"].midbottom, 10, self.COLOR_PARTICLE_JUMP, 2, np.array([0, -0.5]))

        if self.player["jump_cooldown"] > 0:
            self.player["jump_cooldown"] -= 1

    def _update_player(self):
        # Apply friction
        if self.player["vel"][0] > 0:
            self.player["vel"][0] += self.PLAYER_FRICTION
            if self.player["vel"][0] < 0: self.player["vel"][0] = 0
        elif self.player["vel"][0] < 0:
            self.player["vel"][0] -= self.PLAYER_FRICTION
            if self.player["vel"][0] > 0: self.player["vel"][0] = 0
            
        # Apply gravity
        self.player["vel"][1] += self.GRAVITY
        
        # Clamp velocities
        self.player["vel"][0] = np.clip(self.player["vel"][0], -self.PLAYER_MAX_SPEED_X, self.PLAYER_MAX_SPEED_X)
        self.player["vel"][1] = np.clip(self.player["vel"][1], -self.PLAYER_MAX_SPEED_Y, self.PLAYER_MAX_SPEED_Y)
        
        # Update position
        self.player["pos"] += self.player["vel"]
        self.player["rect"].topleft = self.player["pos"]

    def _handle_collisions(self):
        event_reward = 0
        self.player["is_grounded"] = False

        # --- Platform collisions ---
        player_rect = self.player["rect"]
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                # Vertical collision (landing on top)
                if self.player["vel"][1] > 0 and player_rect.bottom < plat.top + self.player["vel"][1] + 1:
                    player_rect.bottom = plat.top
                    self.player["pos"][1] = player_rect.top
                    self.player["vel"][1] = 0
                    self.player["is_grounded"] = True
                # Vertical collision (bumping head)
                elif self.player["vel"][1] < 0 and player_rect.top > plat.bottom + self.player["vel"][1] - 1:
                    player_rect.top = plat.bottom
                    self.player["pos"][1] = player_rect.top
                    self.player["vel"][1] = 0
                # Horizontal collision
                else:
                    if self.player["vel"][0] > 0 and player_rect.right < plat.left + self.player["vel"][0] + 1:
                        player_rect.right = plat.left
                        self.player["pos"][0] = player_rect.left
                        self.player["vel"][0] = 0
                    elif self.player["vel"][0] < 0 and player_rect.left > plat.right + self.player["vel"][0] - 1:
                        player_rect.left = plat.right
                        self.player["pos"][0] = player_rect.left
                        self.player["vel"][0] = 0

        # --- Other collisions ---
        collected_coins = []
        for coin in self.coins:
            if self.player["rect"].colliderect(coin["rect"]):
                collected_coins.append(coin)
                self.score += 1
                event_reward += 10
                # SFX: Coin collect
                self._create_particles(coin["rect"].center, 15, self.COLOR_PARTICLE_COIN, 3)
        self.coins = [c for c in self.coins if c not in collected_coins]

        for obstacle in self.obstacles:
            if self.player["rect"].colliderect(obstacle["rect"]):
                self._lose_life()
                event_reward -= 5
                break # Only process one death per frame

        # --- Fall off screen ---
        if self.player["rect"].top > self.SCREEN_HEIGHT + 50:
            self._lose_life()
            event_reward -= 5
        
        return event_reward
    
    def _lose_life(self):
        self.lives -= 1
        # SFX: Player death
        self._create_particles(self.player["rect"].center, 30, self.COLOR_PARTICLE_DEATH, 4)
        if self.lives > 0:
            self.player["pos"] = np.array([150.0, 200.0])
            self.player["vel"] = np.array([0.0, 0.0])
        else:
            self.game_over = True

    def _check_termination(self):
        return self.lives <= 0 or self.player["rect"].colliderect(self.flag)

    def _generate_level(self):
        self.platforms = []
        self.coins = []
        self.obstacles = []
        
        # --- Difficulty scaling ---
        max_gap = 60 + self.level_num * 5
        min_gap = 20
        max_height_diff = 40 + self.level_num * 10
        obstacle_chance = 0.1 + self.level_num * 0.02

        # Start platform
        x, y = 50, 300
        self.platforms.append(pygame.Rect(x, y, 200, 40))

        # Procedural generation loop
        while x < self.LEVEL_WIDTH_PIXELS:
            gap = self.rng.integers(min_gap, max_gap)
            x += self.platforms[-1].width + gap
            
            y_diff = self.rng.integers(-max_height_diff, max_height_diff)
            y = np.clip(y + y_diff, 150, self.SCREEN_HEIGHT - 50)
            
            width = self.rng.integers(80, 250)
            
            new_plat = pygame.Rect(x, y, width, 40)
            self.platforms.append(new_plat)

            # Add coins above platform
            num_coins = self.rng.integers(1, 5)
            for i in range(num_coins):
                coin_x = new_plat.left + (i + 1) * (new_plat.width / (num_coins + 1))
                self.coins.append({
                    "rect": pygame.Rect(coin_x, new_plat.top - 30, 12, 12),
                    "initial_y": new_plat.top - 30,
                })
            
            # Add obstacles
            if self.rng.random() < obstacle_chance:
                obstacle_x = new_plat.left + new_plat.width / 2
                self.obstacles.append({
                    "pos": np.array([obstacle_x, -20.0]),
                    "vel": np.array([0.0, 1.0 + self.level_num * 0.1]),
                    "rect": pygame.Rect(obstacle_x, -20, 15, 15),
                })
        
        # End flag
        last_plat = self.platforms[-1]
        self.flag = pygame.Rect(last_plat.right - 40, last_plat.top - 50, 30, 50)

    def _update_obstacles(self):
        for obs in self.obstacles:
            obs["pos"] += obs["vel"]
            obs["vel"][1] += self.GRAVITY * 0.2 # Slower gravity for obstacles
            obs["rect"].topleft = obs["pos"]
            # Reset if they fall off screen
            if obs["rect"].top > self.SCREEN_HEIGHT + 100:
                obs["pos"][1] = -20.0
                obs["vel"][1] = 1.0 + self.level_num * 0.1

    def _update_camera(self):
        target_cam_x = self.player["pos"][0] - self.SCREEN_WIDTH / 2
        target_cam_y = self.player["pos"][1] - self.SCREEN_HEIGHT / 2
        
        # Smooth camera movement
        self.camera_offset[0] += (target_cam_x - self.camera_offset[0]) * 0.1
        self.camera_offset[1] += (target_cam_y - self.camera_offset[1]) * 0.1
        
        # Clamp camera to level bounds
        self.camera_offset[0] = max(0, self.camera_offset[0])
        self.camera_offset[0] = min(self.LEVEL_WIDTH_PIXELS - self.SCREEN_WIDTH, self.camera_offset[0])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x, cam_y = int(self.camera_offset[0]), int(self.camera_offset[1])

        # Draw platforms
        for plat in self.platforms:
            if plat.right > cam_x and plat.left < cam_x + self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat.move(-cam_x, -cam_y))

        # Draw coins with animation
        for coin in self.coins:
            if coin["rect"].right > cam_x and coin["rect"].left < cam_x + self.SCREEN_WIDTH:
                # Bobbing and spinning animation
                bob_offset = math.sin(self.steps * 0.05 + coin["rect"].x * 0.1) * 3
                spin_width = (math.sin(self.steps * 0.2) + 1) / 2 * 10 + 2
                
                r = coin["rect"].copy()
                r.y = coin["initial_y"] + bob_offset
                r.width = int(spin_width)
                r.centerx = coin["rect"].centerx
                pygame.draw.ellipse(self.screen, self.COLOR_COIN, r.move(-cam_x, -cam_y))

        # Draw obstacles
        for obs in self.obstacles:
            if obs["rect"].right > cam_x and obs["rect"].left < cam_x + self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs["rect"].move(-cam_x, -cam_y), border_radius=3)

        # Draw flag
        if self.flag.right > cam_x and self.flag.left < cam_x + self.SCREEN_WIDTH:
            pygame.draw.rect(self.screen, self.COLOR_FLAG, self.flag.move(-cam_x, -cam_y))

        # Draw particles
        self._render_particles(cam_x, cam_y)

        # Draw player
        if self.lives > 0:
            player_screen_rect = self.player["rect"].move(-cam_x, -cam_y)
            
            # Simple run/jump animation
            if not self.player["is_grounded"]:
                # Squish when jumping/falling
                anim_rect = player_screen_rect.inflate(-4, 4)
                anim_rect.midbottom = player_screen_rect.midbottom
            else:
                # Bob while running
                bob = abs(math.sin(self.steps * 0.3)) * -4 if abs(self.player["vel"][0]) > 0.1 else 0
                anim_rect = player_screen_rect.move(0, bob)
            
            # Glow effect
            glow_surf = pygame.Surface((anim_rect.width*2, anim_rect.height*2), pygame.SRCALPHA)
            pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect())
            self.screen.blit(glow_surf, (anim_rect.centerx - anim_rect.width, anim_rect.centery - anim_rect.height), special_flags=pygame.BLEND_RGBA_ADD)

            pygame.draw.rect(self.screen, self.COLOR_PLAYER, anim_rect, border_radius=4)

    def _render_ui(self):
        lives_text = self.font.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        score_text = self.font.render(f"COINS: {self.score}", True, self.COLOR_TEXT)
        
        self.screen.blit(lives_text, (10, 10))
        self.screen.blit(score_text, (10, 40))

        if self.game_over and self.lives <= 0:
            game_over_text = self.font.render("GAME OVER", True, self.COLOR_PLAYER)
            text_rect = game_over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "level": self.level_num
        }

    def _create_particles(self, pos, count, color, speed_factor, base_vel=np.array([0.0, 0.0])):
        for _ in range(count):
            angle = self.rng.random() * 2 * math.pi
            speed = self.rng.random() * speed_factor
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed]) + base_vel
            self.particles.append({
                "pos": np.array(pos, dtype=float),
                "vel": vel,
                "lifespan": self.rng.integers(15, 30),
                "color": color,
                "size": self.rng.integers(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"][1] += self.GRAVITY * 0.1 # Particles are affected by gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        
    def _render_particles(self, cam_x, cam_y):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0] - cam_x), int(p["pos"][1] - cam_y))
            size = int(p["size"] * (p["lifespan"] / 30))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

    def close(self):
        pygame.quit()

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy' # Run headless
    
    env = GameEnv()
    obs, info = env.reset()
    
    print("Initial Info:", info)
    
    total_reward = 0
    terminated = False
    
    for i in range(1000):
        action = env.action_space.sample() # Take random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if (i+1) % 100 == 0:
            print(f"Step {i+1}: Info={info}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")

        if terminated:
            print(f"Episode finished after {i+1} steps. Final Info:", info)
            total_reward = 0
            obs, info = env.reset()

    env.close()
    print("Example run completed.")