
# Generated: 2025-08-28T03:23:06.492500
# Source Brief: brief_02003.md
# Brief Index: 2003

        
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
        "Controls: Arrow keys to move on the isometric grid. "
        "Your goal is to collect the blue crystals while avoiding the red traps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game on an isometric grid. "
        "Navigate your avatar to collect 100 crystals to win, but be careful not to fall into the traps!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and Grid Dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.TILE_WIDTH, self.TILE_HEIGHT = 30, 15

        # Game Parameters
        self.WIN_SCORE = 100
        self.MAX_STEPS = 1000
        self.NUM_TRAPS = 40
        self.NUM_CRYSTALS = 120

        # Colors
        self.COLOR_BG = (40, 42, 54)
        self.COLOR_TILE = (50, 52, 64)
        self.COLOR_PLAYER = (80, 250, 123)
        self.COLOR_PLAYER_OUTLINE = (248, 248, 242)
        self.COLOR_CRYSTAL = (139, 233, 253)
        self.COLOR_CRYSTAL_OUTLINE = (248, 248, 242)
        self.COLOR_TRAP = (255, 85, 85)
        self.COLOR_TRAP_OUTLINE = (255, 121, 198)
        self.COLOR_TEXT = (248, 248, 242)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)

        # Centering offset for the grid
        self.grid_offset_x = self.SCREEN_WIDTH / 2
        self.grid_offset_y = self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT / 2.5)

        # Initialize state variables
        self.player_pos = None
        self.traps = None
        self.crystals = None
        self.particles = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None

        self.reset()
        
        # This will fail if the implementation is incorrect.
        self.validate_implementation()

    def _iso_to_screen(self, x, y):
        """Converts isometric grid coordinates to screen coordinates."""
        screen_x = (x - y) * self.TILE_WIDTH / 2 + self.grid_offset_x
        screen_y = (x + y) * self.TILE_HEIGHT / 2 + self.grid_offset_y
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Generate grid entities
        all_coords = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(all_coords)

        # Place player
        self.player_pos = list(all_coords.pop())

        # Place traps
        self.traps = set(all_coords[:self.NUM_TRAPS])
        
        # Place crystals
        available_for_crystals = all_coords[self.NUM_TRAPS:]
        self.crystals = set(available_for_crystals[:self.NUM_CRYSTALS])

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        
        if not self.game_over:
            # Store old position for collision checks
            next_pos = self.player_pos.copy()

            # Isometric movement
            if movement == 1:  # Up
                next_pos[0] -= 1
                next_pos[1] -= 1
            elif movement == 2:  # Down
                next_pos[0] += 1
                next_pos[1] += 1
            elif movement == 3:  # Left
                next_pos[0] -= 1
                next_pos[1] += 1
            elif movement == 4:  # Right
                next_pos[0] += 1
                next_pos[1] -= 1

            # Update player position if within bounds
            if 0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT:
                self.player_pos = next_pos

            # Check for events at the new position
            player_pos_tuple = tuple(self.player_pos)
            
            # Check for trap collision
            if player_pos_tuple in self.traps:
                # sfx: player_death_sfx()
                self.game_over = True
                reward = -100
                screen_pos = self._iso_to_screen(*self.player_pos)
                self._spawn_particles(screen_pos, self.COLOR_TRAP, 30, 3, 40)

            # Check for crystal collection
            elif player_pos_tuple in self.crystals:
                # sfx: crystal_collect_sfx()
                self.crystals.remove(player_pos_tuple)
                self.score += 1
                reward = 1
                screen_pos = self._iso_to_screen(*self.player_pos)
                self._spawn_particles(screen_pos, self.COLOR_CRYSTAL, 15, 2, 30)

                # Check for win condition
                if self.score >= self.WIN_SCORE:
                    # sfx: win_jingle_sfx()
                    self.game_over = True
                    reward += 100

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over

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
        # Draw floor tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                self._draw_iso_poly(x, y, self.COLOR_TILE)

        # Draw traps
        for x, y in self.traps:
            self._draw_iso_poly(x, y, self.COLOR_TRAP, self.COLOR_TRAP_OUTLINE)

        # Draw crystals
        for x, y in self.crystals:
            self._draw_iso_poly(x, y, self.COLOR_CRYSTAL, self.COLOR_CRYSTAL_OUTLINE)
            center_x, center_y = self._iso_to_screen(x,y)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER_OUTLINE, (center_x, center_y), 2)


        # Draw player if not dead
        if not (tuple(self.player_pos) in self.traps):
            player_x, player_y = self._iso_to_screen(*self.player_pos)
            radius = int(self.TILE_WIDTH / 2.5)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, (player_x, player_y), radius)
            pygame.gfxdraw.aacircle(self.screen, player_x, player_y, radius, self.COLOR_PLAYER_OUTLINE)
        
        # Update and draw particles
        self._update_particles()

    def _draw_iso_poly(self, x, y, color, outline_color=None):
        """Draws an anti-aliased isometric polygon."""
        center_x, center_y = self._iso_to_screen(x, y)
        points = [
            (center_x, center_y - self.TILE_HEIGHT / 2),
            (center_x + self.TILE_WIDTH / 2, center_y),
            (center_x, center_y + self.TILE_HEIGHT / 2),
            (center_x - self.TILE_WIDTH / 2, center_y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(self.screen, points, outline_color)


    def _render_ui(self):
        # UI background
        ui_bg = pygame.Surface((self.SCREEN_WIDTH, 35), pygame.SRCALPHA)
        ui_bg.fill((20, 20, 20, 180))
        self.screen.blit(ui_bg, (0, 0))

        # Score text
        score_text = f"CRYSTALS: {self.score} / {self.WIN_SCORE}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 5))

        # Steps text
        steps_text = f"STEPS: {self.steps} / {self.MAX_STEPS}"
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.SCREEN_WIDTH - steps_surf.get_width() - 10, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "crystals_remaining": len(self.crystals),
        }

    def _spawn_particles(self, pos, color, count, max_speed, max_life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(max_life // 2, max_life)
            self.particles.append({
                'pos': list(pos),
                'vel': velocity,
                'life': life,
                'max_life': life,
                'color': color,
            })

    def _update_particles(self):
        for i in range(len(self.particles) - 1, -1, -1):
            p = self.particles[i]
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

            if p['life'] <= 0:
                self.particles.pop(i)
            else:
                # Fade out effect
                alpha = int(255 * (p['life'] / p['max_life']))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2, 2), 2)
                self.screen.blit(temp_surf, (int(p['pos'][0]) - 2, int(p['pos'][1]) - 2))

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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a different screen for display if running manually
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Crystal Collector")
    clock = pygame.time.Clock()

    print(env.user_guide)
    
    total_reward = 0
    total_steps = 0
    
    while True:
        # Convert observation back to a surface for display
        display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(display_surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}, Steps: {total_steps}")
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            total_steps = 0

        action = [0, 0, 0] # Default to no-op
        
        # Pygame event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                
                # Only step if a key was pressed, since auto_advance is False
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_steps += 1

        clock.tick(30) # Limit FPS for manual play