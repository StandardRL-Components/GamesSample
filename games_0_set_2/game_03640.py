
# Generated: 2025-08-27T23:57:55.183627
# Source Brief: brief_03640.md
# Brief Index: 3640

        
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
        "Controls: Arrow keys to move. Avoid the blue zombies and collect yellow "
        "supplies. Reach the green exit to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down survival game. You are the red circle, trying to escape a horde of "
        "blue zombies. Collect yellow supplies for points and reach the green exit to win. "
        "You lose if you are caught 5 times."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 32, 20
        self.TILE_SIZE = 20
        
        self.NUM_ZOMBIES = 5
        self.NUM_SUPPLIES = 3
        self.MAX_LIVES = 5
        self.MAX_STEPS = 1000

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_ZOMBIE = (50, 150, 255)
        self.COLOR_SUPPLY = (255, 220, 50)
        self.COLOR_EXIT = (50, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GAMEOVER_BG = (0, 0, 0, 180)

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
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 60, bold=True)
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.zombie_pos = []
        self.supply_pos = []
        self.exit_pos = [0, 0]
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.game_outcome = ""

        # Initialize state
        self.reset()

        # Run self-check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.game_outcome = ""
        self.particles = []

        self._place_entities()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _place_entities(self):
        """Procedurally places player, exit, supplies, and zombies without overlap."""
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)
        
        self.player_pos = list(all_coords.pop())
        self.exit_pos = list(all_coords.pop())
        
        self.supply_pos = [list(all_coords.pop()) for _ in range(self.NUM_SUPPLIES)]
        self.zombie_pos = [list(all_coords.pop()) for _ in range(self.NUM_ZOMBIES)]

    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = -0.1  # Cost for each step to encourage efficiency
        
        # --- Update game logic ---
        self.steps += 1
        
        # 1. Move Player
        old_player_pos = self.player_pos[:]
        if movement == 1:  # Up
            self.player_pos[1] -= 1
        elif movement == 2:  # Down
            self.player_pos[1] += 1
        elif movement == 3:  # Left
            self.player_pos[0] -= 1
        elif movement == 4:  # Right
            self.player_pos[0] += 1
        
        # Clamp player position to grid
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_WIDTH - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_HEIGHT - 1)

        # Reward for moving closer to exit
        dist_before = math.dist(old_player_pos, self.exit_pos)
        dist_after = math.dist(self.player_pos, self.exit_pos)
        if dist_after < dist_before:
            reward += 0.5
        
        # 2. Move Zombies
        for i in range(len(self.zombie_pos)):
            move = self.np_random.choice(5)
            if move == 1: self.zombie_pos[i][1] -= 1 # Up
            elif move == 2: self.zombie_pos[i][1] += 1 # Down
            elif move == 3: self.zombie_pos[i][0] -= 1 # Left
            elif move == 4: self.zombie_pos[i][0] += 1 # Right
            
            # Clamp zombie position
            self.zombie_pos[i][0] = np.clip(self.zombie_pos[i][0], 0, self.GRID_WIDTH - 1)
            self.zombie_pos[i][1] = np.clip(self.zombie_pos[i][1], 0, self.GRID_HEIGHT - 1)

        # 3. Check Collisions
        # Player <-> Supplies
        for supply in self.supply_pos[:]:
            if self.player_pos == supply:
                self.supply_pos.remove(supply)
                self.score += 10
                reward += 10
                # sfx: supply_pickup.wav
        
        # Player <-> Zombies
        hit_this_step = False
        for zombie in self.zombie_pos:
            if self.player_pos == zombie and not hit_this_step:
                self.lives -= 1
                hit_this_step = True
                self._create_hit_particles(self.player_pos)
                # sfx: player_hit.wav

        # 4. Update particles
        self._update_particles()
        
        # 5. Check Termination Conditions
        terminated = False
        if self.player_pos == self.exit_pos:
            terminated = True
            self.game_over = True
            self.game_outcome = "YOU WIN!"
            reward += 100
            # sfx: win_sound.wav
        elif self.lives <= 0:
            terminated = True
            self.game_over = True
            self.game_outcome = "GAME OVER"
            reward -= 100
            # sfx: lose_sound.wav
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.game_outcome = "TIME UP"
        
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
        self._render_grid()
        self._render_exit()
        self._render_supplies()
        self._render_zombies()
        self._render_player()
        self._render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }
        
    def _grid_to_pixel(self, grid_pos):
        """Converts grid coordinates to pixel coordinates (center of the tile)."""
        px = int(grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE / 2)
        py = int(grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE / 2)
        return px, py

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_exit(self):
        px, py = self._grid_to_pixel(self.exit_pos)
        rect = pygame.Rect(px - self.TILE_SIZE // 2, py - self.TILE_SIZE // 2, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, rect)

    def _render_supplies(self):
        radius = self.TILE_SIZE // 3
        for pos in self.supply_pos:
            px, py = self._grid_to_pixel(pos)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_SUPPLY)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_SUPPLY)

    def _render_zombies(self):
        radius = int(self.TILE_SIZE * 0.4)
        for pos in self.zombie_pos:
            px, py = self._grid_to_pixel(pos)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_ZOMBIE)

    def _render_player(self):
        radius = int(self.TILE_SIZE * 0.45)
        px, py = self._grid_to_pixel(self.player_pos)
        
        # Glow effect
        glow_radius = int(radius * 1.5)
        glow_color = (*self.COLOR_PLAYER, 50) # RGBA
        temp_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(temp_surf, glow_radius, glow_radius, glow_radius, glow_color)
        self.screen.blit(temp_surf, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Player circle
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER)

    def _create_hit_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(10, 20)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({
                "pos": [px, py], "vel": vel, "life": life, "max_life": life, "radius": radius
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            radius = int(p["radius"] * life_ratio)
            color = (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], int(255 * life_ratio))
            
            if radius > 0:
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
                self.screen.blit(temp_surf, (int(p["pos"][0] - radius), int(p["pos"][1] - radius)))

    def _render_ui(self):
        # UI Text
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - lives_text.get_width() - 10, 10))

        # Game Over Screen
        if self.game_over:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill(self.COLOR_GAMEOVER_BG)
            self.screen.blit(s, (0, 0))
            
            outcome_text = self.font_large.render(self.game_outcome, True, self.COLOR_TEXT)
            text_rect = outcome_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(outcome_text, text_rect)

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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0]  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        # Pygame uses (width, height), numpy uses (height, width)
        # Transpose is needed to convert from gym's (H, W, C) to pygame's (W, H, C) representation
        frame = np.transpose(obs, (1, 0, 2))
        
        # Create a pygame surface from the numpy array
        surf = pygame.surfarray.make_surface(frame)
        
        # Display the surface
        # The main script needs its own screen to display the env's rendered surface
        if 'display_screen' not in locals():
            display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
            pygame.display.set_caption("Zombie Survival")

        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
    
    env.close()