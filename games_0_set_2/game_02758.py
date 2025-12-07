
# Generated: 2025-08-27T21:20:57.914761
# Source Brief: brief_02758.md
# Brief Index: 2758

        
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
    A grid-based action game where the player defeats waves of monsters.
    The player must use strategic movement and attacks to clear three waves of enemies
    while avoiding contact, which costs a life.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space and press an arrow key to fire in that direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of monsters on a grid-based battlefield using strategic movement and attacks to survive."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.W, self.H = 640, 400
        self.GRID_SIZE = 8
        self.CELL_SIZE = 40
        self.MAX_STEPS = 1000
        self.MAX_WAVES = 3
        self.MONSTERS_PER_WAVE = 5
        self.STARTING_LIVES = 3
        
        self.GRID_ORIGIN_X = (self.W - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_ORIGIN_Y = (self.H - self.GRID_SIZE * self.CELL_SIZE) // 2

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (50, 255, 150, 50)
        self.COLOR_MONSTER = (255, 80, 80)
        self.COLOR_MONSTER_GLOW = (255, 80, 80, 50)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_OVERLAY = (0, 0, 0, 150)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.monsters = []
        self.projectiles = []
        self.particles = []
        self.current_wave = 0
        self.monster_move_prob = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE - 1]
        self.player_lives = self.STARTING_LIVES
        self.current_wave = 1
        self.monster_move_prob = 0.25
        
        self.monsters = []
        self.projectiles = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        # shift_held is not used in this design but is part of the action space

        # Action logic: Move or Attack
        if space_held and movement in [1, 2, 3, 4]: # Attack
            # Sound: player_shoot.wav
            attack_dir_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            direction = attack_dir_map[movement]
            self.projectiles.append({
                "pos": list(self.player_pos),
                "dir": direction
            })
        elif not space_held and movement in [1, 2, 3, 4]: # Move
            move_dir_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
            dx, dy = move_dir_map[movement]
            new_x = max(0, min(self.GRID_SIZE - 1, self.player_pos[0] + dx))
            new_y = max(0, min(self.GRID_SIZE - 1, self.player_pos[1] + dy))
            self.player_pos = [new_x, new_y]
        else: # No-op
            reward -= 0.1

        # --- Update Game State ---
        
        # Update projectiles
        projectiles_to_remove = []
        monsters_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['pos'][0] += p['dir'][0]
            p['pos'][1] += p['dir'][1]

            # Check for monster collision
            hit = False
            for j, m_pos in enumerate(self.monsters):
                if p['pos'] == m_pos and j not in monsters_to_remove:
                    # Sound: monster_hit.wav
                    reward += 1.0
                    self._create_explosion(m_pos, self.COLOR_MONSTER, 20)
                    monsters_to_remove.append(j)
                    projectiles_to_remove.append(i)
                    hit = True
                    break
            
            if hit: continue

            # Check for out of bounds
            if not (0 <= p['pos'][0] < self.GRID_SIZE and 0 <= p['pos'][1] < self.GRID_SIZE):
                projectiles_to_remove.append(i)
        
        # Remove hit monsters and used projectiles
        self.monsters = [m for i, m in enumerate(self.monsters) if i not in monsters_to_remove]
        self.projectiles = [p for i, p in enumerate(self.projectiles) if i not in projectiles_to_remove]

        # Update monsters
        for i in range(len(self.monsters)):
            if self.np_random.random() < self.monster_move_prob:
                move_options = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dx, dy = self.np_random.choice(move_options)
                new_x = max(0, min(self.GRID_SIZE - 1, self.monsters[i][0] + dx))
                new_y = max(0, min(self.GRID_SIZE - 1, self.monsters[i][1] + dy))
                self.monsters[i] = [new_x, new_y]

        # Check for player-monster collision
        monster_collided_idx = -1
        for i, m_pos in enumerate(self.monsters):
            if self.player_pos == m_pos:
                # Sound: player_damage.wav
                reward -= 1.0
                self.player_lives -= 1
                self._create_explosion(self.player_pos, self.COLOR_PLAYER, 30)
                monster_collided_idx = i
                break
        
        if monster_collided_idx != -1:
            # Remove collided monster and reset player position
            self.monsters.pop(monster_collided_idx)
            self.player_pos = [self.GRID_SIZE // 2, self.GRID_SIZE - 1]

        # Check for wave clear
        if not self.monsters and not self.game_over:
            reward += 10.0
            if self.current_wave >= self.MAX_WAVES:
                # Game Win
                reward += 100.0
                self.game_over = True
            else:
                # Sound: wave_clear.wav
                self.current_wave += 1
                self.monster_move_prob += 0.05
                self._spawn_wave()

        self.steps += 1
        self.score += reward
        
        # Check termination conditions
        terminated = (self.player_lives <= 0) or self.game_over or (self.steps >= self.MAX_STEPS)
        if self.player_lives <= 0 and not self.game_over:
            reward -= 100.0 # Apply terminal penalty
            self.score -= 100.0
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_wave(self):
        occupied_positions = [self.player_pos] + self.monsters
        for _ in range(self.MONSTERS_PER_WAVE):
            while True:
                pos = [self.np_random.integers(0, self.GRID_SIZE), self.np_random.integers(0, self.GRID_SIZE // 2)]
                dist_to_player = abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1])
                if pos not in occupied_positions and dist_to_player > 2:
                    self.monsters.append(pos)
                    occupied_positions.append(pos)
                    break

    def _grid_to_pixel(self, x, y):
        px = self.GRID_ORIGIN_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_ORIGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _create_explosion(self, pos, color, count):
        px, py = self._grid_to_pixel(*pos)
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [px, py],
                "vel": vel,
                "life": self.np_random.uniform(10, 20),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Grid
        for i in range(self.GRID_SIZE + 1):
            start_x, start_y = self._grid_to_pixel(i - 0.5, -0.5)
            end_x, end_y = self._grid_to_pixel(i - 0.5, self.GRID_SIZE - 0.5)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y), 1)
            start_x, start_y = self._grid_to_pixel(-0.5, i - 0.5)
            end_x, end_y = self._grid_to_pixel(self.GRID_SIZE - 0.5, i - 0.5)
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x, start_y), (end_x, end_y), 1)

        # Update and Render Particles
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
            else:
                alpha = int(255 * (p['life'] / 20))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha))
        self.particles = [p for i, p in enumerate(self.particles) if i not in particles_to_remove]

        # Render Monsters
        for pos in self.monsters:
            px, py = self._grid_to_pixel(*pos)
            size = self.CELL_SIZE // 3
            pygame.gfxdraw.filled_circle(self.screen, px, py, size, self.COLOR_MONSTER)
            pygame.gfxdraw.aacircle(self.screen, px, py, size, self.COLOR_MONSTER)
            pygame.gfxdraw.filled_circle(self.screen, px, py, size * 2, self.COLOR_MONSTER_GLOW)


        # Render Projectiles
        for p in self.projectiles:
            px, py = self._grid_to_pixel(*p['pos'])
            size = self.CELL_SIZE // 8
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, (px - size, py - size, size * 2, size * 2))

        # Render Player
        px, py = self._grid_to_pixel(*self.player_pos)
        size = self.CELL_SIZE // 3
        pygame.gfxdraw.filled_circle(self.screen, px, py, size, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, py, size, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, size * 2, self.COLOR_PLAYER_GLOW)

    def _render_ui(self):
        # Score and Lives
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = self.font_small.render(f"LIVES: {self.player_lives}", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, 30))

        # Wave counter
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.W - wave_text.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            if self.player_lives > 0:
                msg = "YOU WIN!"
                color = self.COLOR_PLAYER
            else:
                msg = "GAME OVER"
                color = self.COLOR_MONSTER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.W // 2, self.H // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.W, env.H))
    pygame.display.set_caption("Grid Combat")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # --- Action mapping for human play ---
        movement = 0 # 0=none
        space_held = 0 # 0=released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        
        action = [movement, space_held, 0] # shift is not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
        
        # --- Step the environment ---
        # Since auto_advance is False, we only step on an action.
        # For human play, we can step every frame to feel responsive.
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
            
        if terminated:
            print("Episode finished. Press 'R' to reset.")

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # In a real game loop, you'd want to control the step rate.
        # Here we just sync to a display FPS.
        clock.tick(10) # Slower tick rate for turn-based feel
        
    env.close()