
# Generated: 2025-08-28T05:25:42.323750
# Source Brief: brief_02617.md
# Brief Index: 2617

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space on a plot to plant/harvest. "
        "Hold Shift in the barn to sell crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small farm by planting, harvesting, and selling crops to reach "
        "a target profit within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (139, 172, 15)
    COLOR_PLOT = (92, 53, 15)
    COLOR_PLAYER = (255, 223, 0)
    COLOR_BARN = (192, 57, 43)
    COLOR_SEED = (46, 139, 87)
    COLOR_RIPE = (253, 233, 16)
    COLOR_RIPE_GLOW = (255, 255, 150, 100) # with alpha
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128) # with alpha

    # Game parameters
    WIN_GOLD = 1000
    MAX_STEPS = 600 # 60 seconds at 10 steps/sec
    PLAYER_SPEED = 12
    CROP_SALE_VALUE = 25
    
    # Crop growth stages (in steps)
    GROWTH_TIME = 150

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        
        # Etc...        
        
        # Initialize state variables
        self.player_pos = None
        self.plots = None
        self.inventory = None
        self.gold = None
        self.time_left = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.barn_rect = None

        self.reset()
        
        # Optional: Validate implementation at the end of __init__
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]
        
        self.plots = []
        plot_width, plot_height = 60, 60
        gap = 15
        grid_cols, grid_rows = 6, 3
        grid_width = grid_cols * (plot_width + gap) - gap
        grid_height = grid_rows * (plot_height + gap) - gap
        start_x = (self.SCREEN_WIDTH - grid_width) / 2
        start_y = (self.SCREEN_HEIGHT - grid_height) / 2 + 30

        for row in range(grid_rows):
            for col in range(grid_cols):
                x = start_x + col * (plot_width + gap)
                y = start_y + row * (plot_height + gap)
                self.plots.append({
                    "rect": pygame.Rect(x, y, plot_width, plot_height),
                    "state": "empty", # "empty", "growing", "ripe"
                    "growth_timer": 0
                })
        
        self.barn_rect = pygame.Rect(self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 120, 80, 100)

        self.inventory = {"crops": 0}
        self.gold = 0
        self.time_left = self.MAX_STEPS
        self.steps = 0
        self.game_over = False
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        self.steps += 1
        self.time_left -= 1
        reward = 0

        # 1. Handle Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)
        
        # 2. Update Crop Growth
        for plot in self.plots:
            if plot["state"] == "growing":
                plot["growth_timer"] += 1
                if plot["growth_timer"] >= self.GROWTH_TIME:
                    plot["state"] = "ripe"
                    # sound: crop ready!
                    self._create_particles(plot["rect"].center, 5, self.COLOR_RIPE, 0.5)

        # 3. Handle Interactions
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)
        
        action_taken = False
        if space_held:
            for plot in self.plots:
                if plot["rect"].colliderect(player_rect):
                    if plot["state"] == "empty":
                        plot["state"] = "growing"
                        plot["growth_timer"] = 0
                        action_taken = True
                        # sound: plant seed
                        self._create_particles(plot["rect"].center, 10, self.COLOR_SEED, 0.3)
                        break
                    elif plot["state"] == "ripe":
                        plot["state"] = "empty"
                        self.inventory["crops"] += 1
                        reward += 0.1
                        action_taken = True
                        # sound: harvest
                        self._create_particles(plot["rect"].center, 20, self.COLOR_RIPE, 1, fly_to=self.player_pos)
                        break
        
        if shift_held:
            if self.barn_rect.colliderect(player_rect) and self.inventory["crops"] > 0:
                earned_gold = self.inventory["crops"] * self.CROP_SALE_VALUE
                self.gold += earned_gold
                # Reward for selling, scaled by number of crops
                reward += 1.0 + self.inventory["crops"] * 0.5
                action_taken = True
                # sound: cha-ching!
                self._create_particles(self.player_pos, self.inventory["crops"] * 2, self.COLOR_PLAYER, 1.5, fly_to=(60, 30))
                self.inventory["crops"] = 0

        # No-op penalty
        if not action_taken and movement == 0:
            reward -= 0.01

        # 4. Check Termination
        terminated = self.time_left <= 0 or self.gold >= self.WIN_GOLD
        if terminated:
            self.game_over = True
            if self.gold >= self.WIN_GOLD:
                reward += 100 # Win bonus
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
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
    
    def _render_game(self):
        # Draw Barn
        pygame.draw.rect(self.screen, self.COLOR_BARN, self.barn_rect)
        pygame.draw.rect(self.screen, tuple(c*0.8 for c in self.COLOR_BARN), self.barn_rect, 4)
        barn_text = self.font_small.render("SELL", True, self.COLOR_UI_TEXT)
        self.screen.blit(barn_text, barn_text.get_rect(center=self.barn_rect.center))

        # Draw Plots and Crops
        for plot in self.plots:
            pygame.draw.rect(self.screen, self.COLOR_PLOT, plot["rect"], border_radius=5)
            if plot["state"] == "growing":
                progress = plot["growth_timer"] / self.GROWTH_TIME
                radius = int(2 + progress * (plot["rect"].width / 2 - 5))
                pygame.draw.circle(self.screen, self.COLOR_SEED, plot["rect"].center, radius)
            elif plot["state"] == "ripe":
                center = plot["rect"].center
                glow_radius = int(plot["rect"].width / 2 * 1.1)
                crop_radius = int(plot["rect"].width / 2 * 0.8)
                
                # Draw glow using gfxdraw for alpha blending
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], glow_radius, self.COLOR_RIPE_GLOW)
                pygame.draw.circle(self.screen, self.COLOR_RIPE, center, crop_radius)
        
        # Update and Draw Particles
        self._update_and_draw_particles()

        # Draw Player
        player_size = 12 + math.sin(self.steps * 0.2) * 2 # Bobbing effect
        player_rect = pygame.Rect(0, 0, player_size*2, player_size*2)
        player_rect.center = (int(self.player_pos[0]), int(self.player_pos[1]))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_rect.center, int(player_size))
        pygame.draw.circle(self.screen, (255,255,255), player_rect.center, int(player_size), 2)


    def _render_ui(self):
        # UI Background panels
        s = pygame.Surface((140, 40), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (10, 10))

        s2 = pygame.Surface((120, 40), pygame.SRCALPHA)
        s2.fill(self.COLOR_UI_BG)
        self.screen.blit(s2, (self.SCREEN_WIDTH - 130, 10))

        # Gold Display
        gold_text = self.font_large.render(f"${self.gold}", True, self.COLOR_PLAYER)
        self.screen.blit(gold_text, (20, 12))
        
        # Time Display
        time_str = f"{self.time_left // 10:02d}:{self.time_left % 10 * 6:02d}"
        time_text = self.font_large.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - 120, 12))

        # Crops in inventory
        if self.inventory["crops"] > 0:
            crop_count_text = self.font_small.render(f"Crops: {self.inventory['crops']}", True, self.COLOR_UI_TEXT)
            text_rect = crop_count_text.get_rect(center=(int(self.player_pos[0]), int(self.player_pos[1] - 25)))
            self.screen.blit(crop_count_text, text_rect)

        # Game Over Message
        if self.game_over:
            s_gameover = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s_gameover.fill((0,0,0,180))
            self.screen.blit(s_gameover, (0,0))
            
            message = "YOU WIN!" if self.gold >= self.WIN_GOLD else "TIME'S UP!"
            color = (0, 255, 0) if self.gold >= self.WIN_GOLD else (255, 0, 0)
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text, end_rect)

            score_text = self.font_small.render(f"Final Gold: ${self.gold}", True, self.COLOR_UI_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.gold,
            "steps": self.steps,
            "time_left": self.time_left,
            "crops_in_inventory": self.inventory["crops"]
        }
        
    def _create_particles(self, pos, count, color, duration, fly_to=None):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifespan": duration * 30, # 30 fps assumption
                "max_lifespan": duration * 30,
                "color": color,
                "fly_to": fly_to
            })

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                if p["fly_to"]:
                    # Move towards target
                    target_vec = [p["fly_to"][0] - p["pos"][0], p["fly_to"][1] - p["pos"][1]]
                    dist = math.hypot(*target_vec)
                    if dist > 1:
                        target_vec = [c / dist for c in target_vec]
                    
                    # Ease towards target
                    p["vel"][0] = p["vel"][0] * 0.9 + target_vec[0] * 2.5
                    p["vel"][1] = p["vel"][1] * 0.9 + target_vec[1] * 2.5

                p["pos"][0] += p["vel"][0]
                p["pos"][1] += p["vel"][1]
                
                # Fade out
                alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
                color = (*p["color"], alpha)
                
                s = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (2, 2), 2)
                self.screen.blit(s, (int(p["pos"][0]-2), int(p["pos"][1]-2)))

                active_particles.append(p)
        self.particles = active_particles

    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    # This block allows you to run the game directly to test it
    env = GameEnv()
    env.reset()
    
    running = True
    terminated = False
    
    # To store the state of held keys
    keys_held = {
        "up": False, "down": False, "left": False, "right": False,
        "space": False, "shift": False
    }

    print("Starting manual game test.")
    print(GameEnv.user_guide)
    
    # Use a clock to control the frame rate for manual play
    clock = pygame.time.Clock()
    
    # Create a display for manual testing
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Farm Manager")

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    env.reset()
                    terminated = False
                
                # Update held keys
                if event.key == pygame.K_UP: keys_held["up"] = True
                if event.key == pygame.K_DOWN: keys_held["down"] = True
                if event.key == pygame.K_LEFT: keys_held["left"] = True
                if event.key == pygame.K_RIGHT: keys_held["right"] = True
                if event.key == pygame.K_SPACE: keys_held["space"] = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held["shift"] = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_held["up"] = False
                if event.key == pygame.K_DOWN: keys_held["down"] = False
                if event.key == pygame.K_LEFT: keys_held["left"] = False
                if event.key == pygame.K_RIGHT: keys_held["right"] = False
                if event.key == pygame.K_SPACE: keys_held["space"] = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held["shift"] = False

        # Construct action from held keys
        movement_action = 0 # none
        if keys_held["up"]: movement_action = 1
        elif keys_held["down"]: movement_action = 2
        elif keys_held["left"]: movement_action = 3
        elif keys_held["right"]: movement_action = 4
        
        space_action = 1 if keys_held["space"] else 0
        shift_action = 1 if keys_held["shift"] else 0
        
        action = [movement_action, space_action, shift_action]
        
        # Step the environment
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the display
        # The observation is (H, W, C), but pygame blits (W, H) surfaces
        # We can get the surface directly from the env screen
        display_screen.blit(env.screen, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(10) # Run at 10 steps per second for manual play

    env.close()
    print("Game test finished.")