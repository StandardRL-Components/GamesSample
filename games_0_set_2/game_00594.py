
# Generated: 2025-08-27T14:07:29.090645
# Source Brief: brief_00594.md
# Brief Index: 594

        
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
        "Controls: Use ← and → to run, and ↑ to jump. Collect gold coins to increase your score."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style arcade platformer. Navigate a series of procedurally generated platforms, "
        "collecting coins and racing to the finish line before you fall."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (40, 30, 80)
    COLOR_BG_BOTTOM = (80, 60, 140)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLATFORM = (150, 150, 170)
    COLOR_GOAL = (80, 220, 120)
    COLOR_COIN = (255, 223, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SHADOW = (30, 30, 40)

    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Physics
    GRAVITY = 0.5
    PLAYER_JUMP_STRENGTH = -10
    PLAYER_MOVE_SPEED = 4.5
    PLAYER_FRICTION = 0.85
    MAX_VEL_Y = 15

    # Game
    MAX_STEPS = 2000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.on_ground = None
        self.can_jump = None
        self.platforms = None
        self.coins = None
        self.particles = None
        self.camera_x = None
        self.steps = None
        self.score = None
        self.last_player_x = None
        
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.camera_x = 0
        self.particles = []

        self.player_pos = pygame.Vector2(100, self.SCREEN_HEIGHT - 100)
        self.last_player_x = self.player_pos.x
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, 24, 24)
        self.on_ground = False
        self.can_jump = False
        
        self._generate_level()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        """Procedurally generates platforms and coins."""
        self.platforms = []
        self.coins = []
        
        # Starting platform
        start_plat = pygame.Rect(20, self.SCREEN_HEIGHT - 60, 200, 40)
        self.platforms.append(start_plat)
        
        # Level generation
        current_x = start_plat.right
        level_width = self.MAX_STEPS * self.PLAYER_MOVE_SPEED * 0.8 

        while current_x < level_width:
            gap = self.np_random.integers(60, 100)
            plat_width = self.np_random.integers(80, 250)
            plat_y_offset = self.np_random.integers(-40, 40)
            
            plat_x = current_x + gap
            plat_y = min(
                self.SCREEN_HEIGHT - 40, 
                max(200, start_plat.y + plat_y_offset)
            )

            new_plat = pygame.Rect(plat_x, plat_y, plat_width, 150)
            self.platforms.append(new_plat)

            # Add coins above platforms
            if self.np_random.random() < 0.6: # 60% chance to have coins
                num_coins = self.np_random.integers(1, 4)
                for i in range(num_coins):
                    coin_x = new_plat.centerx - (num_coins-1)*15 + i*30
                    coin_y = new_plat.top - 40
                    coin_rect = pygame.Rect(coin_x - 8, coin_y - 8, 16, 16)
                    self.coins.append({
                        "rect": coin_rect, 
                        "active": True,
                        "anim_offset": self.np_random.random() * math.pi * 2
                    })

            current_x = new_plat.right

        # Goal platform
        goal_plat = pygame.Rect(current_x + 100, self.SCREEN_HEIGHT - 100, 100, 80)
        self.platforms.append(goal_plat)


    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Handle Input ---
        if movement == 1 and self.can_jump:  # Up (Jump)
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.can_jump = False
            self.on_ground = False
            # sfx: jump
        
        if movement == 3:  # Left
            self.player_vel.x -= 1.2
        elif movement == 4:  # Right
            self.player_vel.x += 1.2
        
        # --- Update Physics & Player State ---
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_VEL_Y)

        # Apply friction
        self.player_vel.x *= self.PLAYER_FRICTION
        if abs(self.player_vel.x) < 0.1:
            self.player_vel.x = 0
        self.player_vel.x = max(-self.PLAYER_MOVE_SPEED, min(self.PLAYER_MOVE_SPEED, self.player_vel.x))

        # Update position (horizontal)
        self.player_pos.x += self.player_vel.x
        self.player_rect.x = int(self.player_pos.x)
        
        # Update position (vertical) and handle collision
        self.player_pos.y += self.player_vel.y
        self.player_rect.y = int(self.player_pos.y)

        self.on_ground = False
        for plat in self.platforms:
            if self.player_rect.colliderect(plat) and self.player_vel.y >= 0:
                # Check if player was above the platform in the previous frame
                if (self.player_pos.y - self.player_vel.y + self.player_rect.height) <= plat.top + 1:
                    self.player_rect.bottom = plat.top
                    self.player_pos.y = self.player_rect.y
                    self.player_vel.y = 0
                    self.on_ground = True
                    self.can_jump = True
                    break

        # --- Update Game Logic ---
        reward = 0
        terminated = False
        
        # Reward for moving right
        progress = self.player_pos.x - self.last_player_x
        reward += progress * 0.1
        self.last_player_x = self.player_pos.x

        # Coin collection
        for coin in self.coins:
            if coin["active"] and self.player_rect.colliderect(coin["rect"]):
                coin["active"] = False
                self.score += 10
                reward += 5
                # sfx: coin_collect
                self._create_particles(coin["rect"].center, self.COLOR_COIN, 10)

        # Termination conditions
        # 1. Fall off screen
        if self.player_rect.top > self.SCREEN_HEIGHT:
            terminated = True
            reward = -100
            # sfx: fall_death
        
        # 2. Reach goal platform (last platform in the list)
        goal_platform = self.platforms[-1]
        if self.player_rect.colliderect(goal_platform):
            terminated = True
            reward = 100
            self.score += 100
            # sfx: level_complete
            self._create_particles(self.player_rect.center, self.COLOR_GOAL, 30)

        # 3. Max steps
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True

        # --- Update Camera & Particles ---
        target_camera_x = self.player_pos.x - self.SCREEN_WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1
        self._update_particles()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": pygame.Vector2(
                    (self.np_random.random() - 0.5) * 4,
                    (self.np_random.random() - 0.5) * 4 - 1
                ),
                "lifespan": self.np_random.integers(20, 40),
                "color": color,
                "size": self.np_random.integers(3, 7)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["vel"].y += 0.1 # particle gravity
            p["lifespan"] -= 1
        self.particles = [p for p in self.particles if p["lifespan"] > 0]

    def _get_observation(self):
        # Clear screen with background gradient
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        cam_x = int(self.camera_x)
        
        # Draw platforms
        for i, plat in enumerate(self.platforms):
            is_goal = i == len(self.platforms) - 1
            color = self.COLOR_GOAL if is_goal else self.COLOR_PLATFORM
            
            # Shadow
            shadow_rect = plat.move(-cam_x + 4, 4)
            pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow_rect, border_radius=4)
            # Main platform
            main_rect = plat.move(-cam_x, 0)
            pygame.draw.rect(self.screen, color, main_rect, border_radius=4)

        # Draw coins
        for coin in self.coins:
            if coin["active"]:
                anim_phase = (self.steps * 0.2 + coin["anim_offset"]) % (2 * math.pi)
                scale = (math.sin(anim_phase) + 1) / 2 # 0 to 1
                width = int(4 + 12 * scale)
                
                rect = coin["rect"].copy()
                rect.width = width
                rect.centerx = coin["rect"].centerx
                rect.move_ip(-cam_x, 0)
                
                pygame.draw.rect(self.screen, self.COLOR_COIN, rect, border_radius=int(width/2))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["lifespan"] / 40))))
            color = (*p["color"], alpha)
            pos = (int(p["pos"].x - cam_x), int(p["pos"].y))
            size = int(p["size"] * (p["lifespan"] / 40))
            if size > 0:
                # Using gfxdraw for antialiasing
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)

        # Draw player
        player_draw_rect = self.player_rect.move(-cam_x, 0)
        # Shadow
        shadow_rect = player_draw_rect.move(3, 3)
        pygame.draw.rect(self.screen, self.COLOR_SHADOW, shadow_rect, border_radius=3)
        # Main body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_draw_rect, border_radius=3)
        
    def _render_ui(self):
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_text = self.font_large.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_pos.x,
            "player_y": self.player_pos.y,
        }

    def close(self):
        pygame.quit()
        
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
    # Set this to "dummy" to run headlessly
    import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv(render_mode="rgb_array")
    env.reset()
    
    # --- Pygame window for human interaction ---
    if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy":
        pygame.display.set_caption("Platformer Game")
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        
        running = True
        total_reward = 0
        
        # --- Main Game Loop for Human Player ---
        while running:
            # Get keyboard input
            keys = pygame.key.get_pressed()
            
            movement_action = 0 # No-op
            if keys[pygame.K_UP]:
                movement_action = 1
            elif keys[pygame.K_DOWN]:
                movement_action = 2
            elif keys[pygame.K_LEFT]:
                movement_action = 3
            elif keys[pygame.K_RIGHT]:
                movement_action = 4
            
            space_action = 1 if keys[pygame.K_SPACE] else 0
            shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement_action, space_action, shift_action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Handle window events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Render to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
                total_reward = 0
                env.reset()

            env.clock.tick(30) # Control FPS

    env.close()