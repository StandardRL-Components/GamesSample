
# Generated: 2025-08-28T03:09:19.315225
# Source Brief: brief_04836.md
# Brief Index: 4836

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys (↑↓←→) to move your ship and evade the enemy drones."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive for 60 seconds against waves of homing drones in a circular arena. "
        "The longer you last, the more drones appear and the faster they get."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (15, 18, 28)
    COLOR_ARENA_FILL = (25, 30, 45)
    COLOR_ARENA_BORDER = (80, 90, 110)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 50) # RGBA
    COLOR_ZOMBIE_SLOW = (255, 80, 80)
    COLOR_ZOMBIE_MED = (255, 160, 80)
    COLOR_ZOMBIE_FAST = (220, 100, 255)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

    # Game parameters
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4.0
    ZOMBIE_SIZE = 10
    ARENA_CENTER = (WIDTH // 2, HEIGHT // 2)
    ARENA_RADIUS = 180

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_msg = pygame.font.SysFont("sans", 50, bold=True)

        # Initialize state variables to avoid errors
        self.player_pos = [0, 0]
        self.zombies = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win = False
        self.zombie_count = 0
        self.zombie_base_speed = 0.0
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Optional: Call to verify during development

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.player_pos = [self.ARENA_CENTER[0], self.ARENA_CENTER[1]]
        self.zombies = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        # Difficulty state
        self.zombie_count = 3
        self.zombie_base_speed = 1.0
        self._spawn_zombies(self.zombie_count)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            # If the game is over, no more actions, just return the last state
            reward = 0.0
            terminated = True
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                self._get_info(),
            )

        # --- Action Handling ---
        movement = action[0]
        self._move_player(movement)

        # --- Game Logic ---
        self._move_zombies()
        self._update_particles()
        self.steps += 1

        # --- Collision and Termination Check ---
        reward = 0.1  # Survival reward per step
        terminated = False

        if self._check_collision():
            # Player-zombie collision
            reward = -100.0
            terminated = True
            self.game_over = True
            self._create_explosion(self.player_pos, self.COLOR_LOSE, 50)
            # sfx: player_explosion
        elif self.steps >= self.MAX_STEPS:
            # Player survives for 60 seconds
            reward = 100.0
            terminated = True
            self.game_over = True
            self.win = True
            self._create_explosion(self.player_pos, self.COLOR_WIN, 50)
            # sfx: win_sound
        
        # --- Difficulty Scaling ---
        # Every 15 seconds (450 steps)
        if self.steps > 0 and self.steps % (15 * self.FPS) == 0:
            self.zombie_base_speed += 0.2
            new_zombies_to_spawn = 2
            self.zombie_count += new_zombies_to_spawn
            self._spawn_zombies(new_zombies_to_spawn)
            # sfx: new_wave_alert

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
        }

    def _render_game(self):
        # Draw arena
        pygame.gfxdraw.filled_circle(
            self.screen, self.ARENA_CENTER[0], self.ARENA_CENTER[1], self.ARENA_RADIUS, self.COLOR_ARENA_FILL
        )
        pygame.gfxdraw.aacircle(
            self.screen, self.ARENA_CENTER[0], self.ARENA_CENTER[1], self.ARENA_RADIUS, self.COLOR_ARENA_BORDER
        )
        
        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"], int(p["radius"]))

        # Draw zombies
        for z in self.zombies:
            z_rect = pygame.Rect(
                z["pos"][0] - self.ZOMBIE_SIZE / 2,
                z["pos"][1] - self.ZOMBIE_SIZE / 2,
                self.ZOMBIE_SIZE,
                self.ZOMBIE_SIZE
            )
            pygame.draw.rect(self.screen, z["color"], z_rect)

        # Draw player glow and player
        if not self.game_over:
            # Glow effect
            glow_surf = pygame.Surface((self.PLAYER_SIZE * 3, self.PLAYER_SIZE * 3), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (self.PLAYER_SIZE * 1.5, self.PLAYER_SIZE * 1.5), self.PLAYER_SIZE * 1.5)
            self.screen.blit(glow_surf, (self.player_pos[0] - self.PLAYER_SIZE * 1.5, self.player_pos[1] - self.PLAYER_SIZE * 1.5))
            
            # Player
            player_rect = pygame.Rect(
                self.player_pos[0] - self.PLAYER_SIZE / 2,
                self.player_pos[1] - self.PLAYER_SIZE / 2,
                self.PLAYER_SIZE,
                self.PLAYER_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
    
    def _render_ui(self):
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Game Over / Win Message
        if self.game_over:
            if self.win:
                msg_text = self.font_msg.render("SURVIVED!", True, self.COLOR_WIN)
            else:
                msg_text = self.font_msg.render("ELIMINATED", True, self.COLOR_LOSE)
            
            text_rect = msg_text.get_rect(center=self.ARENA_CENTER)
            self.screen.blit(msg_text, text_rect)

    def _move_player(self, movement):
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        self.player_pos[0] += dx * self.PLAYER_SPEED
        self.player_pos[1] += dy * self.PLAYER_SPEED

        # Clamp player to arena
        dist_from_center = math.hypot(self.player_pos[0] - self.ARENA_CENTER[0], self.player_pos[1] - self.ARENA_CENTER[1])
        if dist_from_center > self.ARENA_RADIUS - self.PLAYER_SIZE / 2:
            angle = math.atan2(self.player_pos[1] - self.ARENA_CENTER[1], self.player_pos[0] - self.ARENA_CENTER[0])
            self.player_pos[0] = self.ARENA_CENTER[0] + (self.ARENA_RADIUS - self.PLAYER_SIZE / 2) * math.cos(angle)
            self.player_pos[1] = self.ARENA_CENTER[1] + (self.ARENA_RADIUS - self.PLAYER_SIZE / 2) * math.sin(angle)

    def _move_zombies(self):
        for z in self.zombies:
            dx = self.player_pos[0] - z["pos"][0]
            dy = self.player_pos[1] - z["pos"][1]
            dist = math.hypot(dx, dy)
            if dist > 1: # Avoid division by zero
                dx /= dist
                dy /= dist
            
            z["pos"][0] += dx * z["speed"]
            z["pos"][1] += dy * z["speed"]

    def _spawn_zombies(self, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            spawn_radius = self.ARENA_RADIUS + 30
            x = self.ARENA_CENTER[0] + spawn_radius * math.cos(angle)
            y = self.ARENA_CENTER[1] + spawn_radius * math.sin(angle)
            
            speed_multiplier = self.np_random.uniform(0.9, 1.2)
            speed = self.zombie_base_speed * speed_multiplier
            
            if speed < self.zombie_base_speed * 1.0:
                color = self.COLOR_ZOMBIE_SLOW
            elif speed < self.zombie_base_speed * 1.1:
                color = self.COLOR_ZOMBIE_MED
            else:
                color = self.COLOR_ZOMBIE_FAST

            self.zombies.append({"pos": [x, y], "speed": speed, "color": color})
            # sfx: zombie_spawn

    def _check_collision(self):
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for z in self.zombies:
            zombie_rect = pygame.Rect(z["pos"][0] - self.ZOMBIE_SIZE/2, z["pos"][1] - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)
            if player_rect.colliderect(zombie_rect):
                return True
        return False

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({"pos": list(pos), "vel": vel, "life": life, "radius": radius, "color": color})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            p["radius"] *= 0.95 # Shrink effect
        self.particles = [p for p in self.particles if p["life"] > 0]

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


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # It's a useful way to test and debug the environment
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'x11', 'dummy' or 'windows'
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    
    terminated = False
    
    # --- Human Controls ---
    # Map keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not terminated:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # Get pressed keys for continuous movement
        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                break # Prioritize first key found (e.g., up over down)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait a bit before closing so the player can see the final screen
            pygame.time.wait(3000)
            
    env.close()