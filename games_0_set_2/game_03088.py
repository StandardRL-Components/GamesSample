
# Generated: 2025-08-27T22:19:36.428367
# Source Brief: brief_03088.md
# Brief Index: 3088

        
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
    """
    A procedurally generated platform jumping game where an agent learns to navigate
    increasingly difficult levels to reach a flag. The game features retro pixel-art
    visuals, real-time physics, and a camera that follows the player's vertical movement.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use ←→ to move. Use ↑ for a normal jump, ↓ for a short hop, "
        "space for a high jump, and shift for a long horizontal dash."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated platformer. Jump between platforms to reach "
        "the red flag at the top. Levels get harder as you progress. Don't fall or run out of time!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_PLAYER = (57, 255, 20)  # Bright Green
    COLOR_PLAYER_GLOW = (57, 255, 20, 60)
    COLOR_PLATFORM = (180, 180, 180)
    COLOR_PLATFORM_MOVING = (220, 180, 120)
    COLOR_FLAG = (255, 50, 50)
    COLOR_BG_TOP = (25, 25, 112) # Midnight Blue
    COLOR_BG_BOTTOM = (0, 0, 0) # Black
    COLOR_TEXT = (255, 255, 255)
    COLOR_PARTICLE = (220, 220, 220)

    # Physics
    FPS = 30
    GRAVITY = 0.8
    PLAYER_MOVE_SPEED = 5
    JUMP_NORMAL = 15
    JUMP_SHORT = 8
    JUMP_HIGH = 20
    DASH_STRENGTH = 10
    FRICTION = 0.85
    MAX_VEL = 20

    # Game
    MAX_STEPS = 30 * 60 # 60 seconds at 30fps
    LEVEL_TIME = 60 # seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables
        self.player_pos = None
        self.player_vel = None
        self.on_ground = False
        self.last_platform_landed = None

        self.platforms = []
        self.moving_platforms = []
        self.flag_pos = None
        self.flag_rect = None
        
        self.particles = []
        self.camera_y = 0.0
        
        self.level = 1
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.max_height_reached = 0

        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def _generate_level(self):
        self.platforms = []
        self.moving_platforms = []
        
        # Starting platform
        start_platform = pygame.Rect(self.SCREEN_WIDTH // 2 - 50, 50, 100, 20)
        self.platforms.append(start_platform)
        
        player_start_x = start_platform.centerx
        player_start_y = start_platform.top + 20
        self.player_pos = pygame.Vector2(player_start_x, player_start_y)

        # Procedural generation
        current_y = start_platform.y
        current_x = start_platform.centerx
        
        num_platforms = 15 + self.level * 2
        platform_gap_increase = self.level * 5

        for i in range(num_platforms):
            width = self.np_random.integers(60, 120)
            height = 20
            
            # Ensure next platform is reachable
            max_horiz_dist = self.PLAYER_MOVE_SPEED * 15 + self.DASH_STRENGTH
            max_vert_dist = self.JUMP_HIGH * 8 

            dx = self.np_random.uniform(-max_horiz_dist * 0.6, max_horiz_dist * 0.6)
            dy = self.np_random.uniform(50 + platform_gap_increase, 100 + platform_gap_increase)
            
            next_x = current_x + dx
            next_y = current_y + dy
            
            # Clamp to screen bounds
            next_x = np.clip(next_x, width // 2, self.SCREEN_WIDTH - width // 2)

            is_moving = self.level > 1 and self.np_random.random() < 0.1 * (self.level - 1)
            
            if is_moving:
                move_range = self.np_random.integers(50, 150)
                move_speed = self.np_random.uniform(1.0, 2.0 + self.level * 0.2)
                self.moving_platforms.append({
                    "rect": pygame.Rect(next_x - width // 2, next_y, width, height),
                    "start_x": next_x - width // 2,
                    "range": move_range,
                    "speed": move_speed,
                    "direction": 1
                })
            else:
                self.platforms.append(pygame.Rect(next_x - width // 2, next_y, width, height))

            current_x = next_x
            current_y = next_y

        # Place flag on the last platform
        last_platform = self.moving_platforms[-1]["rect"] if self.moving_platforms and current_y == self.moving_platforms[-1]["rect"].y else self.platforms[-1]
        self.flag_pos = pygame.Vector2(last_platform.centerx, last_platform.top + 30)
        self.flag_rect = pygame.Rect(self.flag_pos.x, self.flag_pos.y - 30, 10, 30)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        if options and 'level' in options:
            self.level = options.get('level', 1)
            self.score = options.get('score', 0)
        else:
            self.level = 1
            self.score = 0

        self.steps = 0
        self.time_remaining = self.LEVEL_TIME * self.FPS
        self.max_height_reached = 0
        self.camera_y = 0
        self.particles = []
        
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.last_platform_landed = -1 # Use index to track

        self._generate_level()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Input ---
        if self.on_ground:
            # Horizontal movement
            if movement == 3: # Left
                self.player_vel.x = -self.PLAYER_MOVE_SPEED
            elif movement == 4: # Right
                self.player_vel.x = self.PLAYER_MOVE_SPEED

            # Jumps (only one type of jump per frame, priority based)
            if space_held: # High jump
                self.player_vel.y = self.JUMP_HIGH
                self.on_ground = False
                # sfx: high_jump
            elif shift_held: # Dash
                direction = math.copysign(1, self.player_vel.x) if self.player_vel.x != 0 else 1
                self.player_vel.x += self.DASH_STRENGTH * direction
                self.on_ground = False
                # sfx: dash
            elif movement == 1: # Normal jump
                self.player_vel.y = self.JUMP_NORMAL
                self.on_ground = False
                # sfx: normal_jump
            elif movement == 2: # Short hop
                self.player_vel.y = self.JUMP_SHORT
                self.on_ground = False
                # sfx: short_jump
        
        # --- Physics Update ---
        # Apply gravity
        self.player_vel.y -= self.GRAVITY
        
        # Apply friction if on ground, air resistance otherwise
        if self.on_ground:
            self.player_vel.x *= self.FRICTION
        else:
            self.player_vel.x *= 0.98 # Air resistance
            reward -= 0.01 # Penalty for being in the air

        # Clamp velocity
        self.player_vel.x = np.clip(self.player_vel.x, -self.MAX_VEL, self.MAX_VEL)
        self.player_vel.y = np.clip(self.player_vel.y, -self.MAX_VEL, self.MAX_VEL)

        # Update position
        self.player_pos += self.player_vel

        # Update moving platforms
        for mp in self.moving_platforms:
            mp["rect"].x += mp["speed"] * mp["direction"]
            if mp["rect"].left < mp["start_x"] or mp["rect"].right > mp["start_x"] + mp["rect"].width + mp["range"]:
                mp["direction"] *= -1

        # --- Collision Detection ---
        self.on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 20, 20, 20)

        all_platforms = self.platforms + [mp["rect"] for mp in self.moving_platforms]
        for i, plat in enumerate(all_platforms):
            if player_rect.colliderect(plat) and self.player_vel.y <= 0:
                # Check if player was above the platform in the last frame
                prev_player_bottom = player_rect.bottom + self.player_vel.y
                if prev_player_bottom <= plat.top:
                    self.player_pos.y = plat.top + 20
                    self.player_vel.y = 0
                    self.on_ground = True
                    # sfx: land
                    if self.last_platform_landed != i:
                        reward += 0.1 # Reward for landing on a new platform
                        self.score += 10
                        self.last_platform_landed = i
                    
                    # Landing particle effect
                    for _ in range(5):
                        self.particles.append({
                            "pos": self.player_pos + pygame.Vector2(random.uniform(-10, 10), -15),
                            "vel": pygame.Vector2(random.uniform(-2, 2), random.uniform(1, 4)),
                            "life": 15
                        })
                    break

        # --- Update Game State ---
        self.steps += 1
        self.time_remaining -= 1

        # Reward for new height
        if self.player_pos.y > self.max_height_reached:
            reward += 1.0
            self.max_height_reached = self.player_pos.y

        # Keep player on screen horizontally
        if self.player_pos.x < 10: self.player_pos.x = 10
        if self.player_pos.x > self.SCREEN_WIDTH - 10: self.player_pos.x = self.SCREEN_WIDTH - 10

        # --- Termination Check ---
        terminated = False
        if self.player_pos.y < self.camera_y - 20: # Fell off bottom
            reward = -10
            self.score -= 50
            terminated = True
        elif self.time_remaining <= 0: # Time ran out
            reward = -10
            self.score -= 50
            terminated = True
        elif player_rect.colliderect(self.flag_rect): # Reached flag
            time_bonus = (self.time_remaining / self.FPS)
            reward = 100 + time_bonus
            self.score += 1000 + int(time_bonus * 10)
            terminated = True
            # Transition to next level
            self.level += 1
            self.reset(options={'level': self.level, 'score': self.score})
            # We return terminated=True, but the internal state is ready for the next level
            # The agent's training loop will call reset() anyway.
        
        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["vel"].y -= 0.2
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        # Update camera
        self.camera_y = max(0, self.player_pos.y - self.SCREEN_HEIGHT * 0.6)

        # Draw background gradient
        for i in range(self.SCREEN_HEIGHT):
            ratio = i / self.SCREEN_HEIGHT
            color = (
                int(self.COLOR_BG_BOTTOM[0] * ratio + self.COLOR_BG_TOP[0] * (1 - ratio)),
                int(self.COLOR_BG_BOTTOM[1] * ratio + self.COLOR_BG_TOP[1] * (1 - ratio)),
                int(self.COLOR_BG_BOTTOM[2] * ratio + self.COLOR_BG_TOP[2] * (1 - ratio))
            )
            pygame.draw.line(self.screen, color, (0, i), (self.SCREEN_WIDTH, i))

        # --- Render Game Elements (with camera offset) ---
        def to_screen_pos(pos):
            return int(pos[0]), int(self.SCREEN_HEIGHT - (pos[1] - self.camera_y))

        # Draw platforms
        for plat in self.platforms:
            screen_rect = plat.move(0, -self.camera_y)
            screen_rect.y = self.SCREEN_HEIGHT - screen_rect.y - screen_rect.height
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)
        
        for mp in self.moving_platforms:
            screen_rect = mp["rect"].move(0, -self.camera_y)
            screen_rect.y = self.SCREEN_HEIGHT - screen_rect.y - screen_rect.height
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_MOVING, screen_rect, border_radius=3)

        # Draw flag
        flag_screen_pos = to_screen_pos(self.flag_pos)
        pygame.draw.line(self.screen, self.COLOR_TEXT, flag_screen_pos, (flag_screen_pos[0], flag_screen_pos[1] + 30), 2)
        pygame.draw.polygon(self.screen, self.COLOR_FLAG, [flag_screen_pos, (flag_screen_pos[0] + 20, flag_screen_pos[1] + 7), (flag_screen_pos[0], flag_screen_pos[1] + 14)])

        # Draw particles
        for p in self.particles:
            p_screen_pos = to_screen_pos(p["pos"])
            alpha = max(0, 255 * (p["life"] / 15))
            size = max(1, int(3 * (p["life"] / 15)))
            pygame.gfxdraw.box(self.screen, (p_screen_pos[0]-size//2, p_screen_pos[1]-size//2, size, size), (*self.COLOR_PARTICLE, alpha))

        # Draw player
        player_screen_pos = to_screen_pos(self.player_pos)
        player_rect = pygame.Rect(player_screen_pos[0] - 10, player_screen_pos[1] - 20, 20, 20)
        
        # Glow effect
        glow_rect = player_rect.inflate(20, 20)
        glow_surface = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, self.COLOR_PLAYER_GLOW, glow_surface.get_rect(), border_radius=10)
        self.screen.blit(glow_surface, glow_rect.topleft)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # --- Render UI Overlay ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Level
        level_text = self.font_small.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (10, 30))

        # Timer
        time_str = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_text = self.font_small.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "player_pos": tuple(self.player_pos),
            "time_remaining": self.time_remaining // self.FPS,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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


if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Platformer RL Environment")
    human_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0
    
    # --- Main Game Loop for Human Play ---
    while running:
        # --- Action mapping for human keyboard ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment...")
                obs, info = env.reset()
                total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Score: {info['score']}, Level: {info['level']}")
            print("Press 'R' to restart.")
            # Wait for reset
            wait_for_reset = True
            while wait_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        wait_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        wait_for_reset = False
                clock.tick(GameEnv.FPS)


        clock.tick(GameEnv.FPS)

    env.close()