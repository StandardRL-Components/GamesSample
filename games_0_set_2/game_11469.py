import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:04:45.113938
# Source Brief: brief_01469.md
# Brief Index: 1469
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a conveyor belt puzzle game.

    The player's goal is to rotate three horizontal conveyor belts to align
    blocks of the same color in the leftmost "matching zone". Matching three
    blocks removes them, awards points, and can trigger chain reaction bonuses.
    The game is score-based and runs for a fixed number of steps.

    **Action Space:** MultiDiscrete([5, 2, 2])
    - `actions[0]` (Movement):
        - 0: No-op
        - 1: Rotate top belt clockwise
        - 2: Rotate bottom belt clockwise
        - 3: Rotate middle belt clockwise
        - 4: No-op
    - `actions[1]` (Space): Unused
    - `actions[2]` (Shift): Unused

    **Observation Space:** Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game state.

    **Rewards:**
    - +1.0 for a 3-block match.
    - +5, +10, +15, ... bonus for consecutive chain reaction matches.
    - +0.01 per step for survival.

    **Termination:**
    - The episode ends after 1000 steps.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Rotate three conveyor belts to align blocks of the same color in the matching zone. "
        "Create combos and chain reactions to maximize your score before time runs out."
    )
    user_guide = (
        "Controls: Use ↑ to rotate the top belt, ↓ for the bottom belt, and ← or → for the middle belt. "
        "Align three blocks of the same color in the left zone to score."
    )
    auto_advance = True

    # Class attribute for persistent high score
    high_score = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Affects game speed and animation smoothness
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_BELT = (60, 70, 90)
        self.COLOR_ZONE = (80, 200, 255, 50) # Matching zone highlight
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_SCORE = (255, 220, 100)
        self.BLOCK_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 150, 255),  # Blue
            (255, 220, 80),  # Yellow
            (200, 80, 255),  # Purple
            (255, 150, 80),  # Orange
        ]
        
        # Game Mechanics
        self.NUM_BELTS = 3
        self.BLOCKS_PER_BELT = 10
        self.BLOCK_SIZE = 40
        self.BLOCK_SPACING = 60
        self.BLOCK_SPEED = 2.0 # pixels per step
        self.CHAIN_REACTION_STEPS = 10 # Max steps between matches to count as a chain

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 20)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 42)
            self.font_small = pygame.font.Font(None, 26)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.belts = []
        self.block_positions = []
        self.particles = []
        self.last_match_step = -self.CHAIN_REACTION_STEPS
        self.chain_count = 0
        self.rotation_feedback = [] # To store (belt_index, alpha) for glow effect

        # self.reset() is called by the wrapper, but we can call it to initialize state
        # self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Update high score if previous game's score was higher
        if hasattr(self, 'score') and self.score > GameEnv.high_score:
            GameEnv.high_score = self.score

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_match_step = -self.CHAIN_REACTION_STEPS
        self.chain_count = 0
        self.particles.clear()
        self.rotation_feedback.clear()
        
        self.belts = [
            [self.np_random.integers(0, len(self.BLOCK_COLORS)) for _ in range(self.BLOCKS_PER_BELT)]
            for _ in range(self.NUM_BELTS)
        ]
        
        # Initialize block x-positions, starting from the right
        self.block_positions = [
            [self.WIDTH - self.BLOCK_SPACING * (i + 1) for i in range(self.BLOCKS_PER_BELT)]
            for _ in range(self.NUM_BELTS)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.01  # Small survival reward per step

        # --- Handle Actions ---
        movement = action[0]
        
        # Map actions to belt rotations
        belt_to_rotate = -1
        if movement == 1: belt_to_rotate = 0 # Up -> Top belt
        if movement == 2: belt_to_rotate = 2 # Down -> Bottom belt
        if movement == 3: belt_to_rotate = 1 # Left -> Middle belt

        if belt_to_rotate != -1:
            # Rotate belt logic: move last block to the front
            self.belts[belt_to_rotate] = [self.belts[belt_to_rotate][-1]] + self.belts[belt_to_rotate][:-1]
            # Add visual feedback for the rotation
            self.rotation_feedback.append([belt_to_rotate, 255])
            # Sound placeholder: pygame.mixer.Sound('rotate.wav').play()

        # --- Update Game State ---
        self._update_block_positions()
        match_reward = self._handle_matches()
        reward += match_reward

        self._update_particles()
        self._update_rotation_feedback()
        
        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info()
        )

    def _update_block_positions(self):
        for i in range(self.NUM_BELTS):
            for j in range(self.BLOCKS_PER_BELT):
                self.block_positions[i][j] -= self.BLOCK_SPEED
            
            # Check if the leftmost block has gone off-screen
            if self.block_positions[i][0] < -self.BLOCK_SIZE:
                # Remove the leftmost block's position
                self.block_positions[i].pop(0)
                # Add a new position for a block entering from the right
                new_pos = self.block_positions[i][-1] + self.BLOCK_SPACING
                self.block_positions[i].append(new_pos)
                
                # Cycle the block color data
                self.belts[i].pop(0)
                self.belts[i].append(self.np_random.integers(0, len(self.BLOCK_COLORS)))

    def _handle_matches(self):
        match_reward = 0
        zone_x = self.BLOCK_SPACING / 2

        # Find which block on each belt is in the matching zone
        zone_blocks = [-1] * self.NUM_BELTS
        for i in range(self.NUM_BELTS):
            for j in range(self.BLOCKS_PER_BELT):
                pos = self.block_positions[i][j]
                if zone_x - self.BLOCK_SIZE / 2 < pos < zone_x + self.BLOCK_SIZE / 2:
                    zone_blocks[i] = j
                    break
        
        # Check for a match if all belts have a block in the zone
        if all(b != -1 for b in zone_blocks):
            color1 = self.belts[0][zone_blocks[0]]
            color2 = self.belts[1][zone_blocks[1]]
            color3 = self.belts[2][zone_blocks[2]]
            
            if color1 == color2 == color3:
                # --- MATCH FOUND ---
                match_reward += 1.0
                self.score += 1

                # Check for chain reaction
                if self.steps - self.last_match_step <= self.CHAIN_REACTION_STEPS:
                    self.chain_count += 1
                else:
                    self.chain_count = 1
                self.last_match_step = self.steps

                # Add chain bonus (starts from the 2nd match in a chain)
                if self.chain_count >= 2:
                    bonus = 5 * (self.chain_count - 1)
                    match_reward += bonus
                    self.score += bonus

                # Create particle explosion
                match_pos_y = self.HEIGHT / 2
                self._create_particles(zone_x, match_pos_y, self.BLOCK_COLORS[color1])
                # Sound placeholder: pygame.mixer.Sound('match.wav').play()

                # Remove matched blocks and shift others
                for i in range(self.NUM_BELTS):
                    block_idx_to_remove = zone_blocks[i]
                    
                    # Remove the block and its position
                    self.belts[i].pop(block_idx_to_remove)
                    self.block_positions[i].pop(block_idx_to_remove)
                    
                    # Shift all blocks to the left of the removed one
                    for k in range(block_idx_to_remove, len(self.block_positions[i])):
                        self.block_positions[i][k] -= self.BLOCK_SPACING

                    # Add a new block at the end
                    self.belts[i].append(self.np_random.integers(0, len(self.BLOCK_COLORS)))
                    new_pos = self.block_positions[i][-1] + self.BLOCK_SPACING
                    self.block_positions[i].append(new_pos)

        return match_reward

    def _create_particles(self, x, y, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(20, 40)
            self.particles.append([x, y, vx, vy, lifetime, color])

    def _update_particles(self):
        self.particles = [
            [p[0] + p[2], p[1] + p[3], p[2] * 0.98, p[3] * 0.98, p[4] - 1, p[5]]
            for p in self.particles if p[4] > 0
        ]
    
    def _update_rotation_feedback(self):
        new_feedback = []
        for feedback in self.rotation_feedback:
            feedback[1] -= 25 # Fade speed
            if feedback[1] > 0:
                new_feedback.append(feedback)
        self.rotation_feedback = new_feedback

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "high_score": GameEnv.high_score,
        }

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # --- Belts and Matching Zone ---
        belt_height = self.BLOCK_SIZE + 20
        belt_y_positions = [
            self.HEIGHT / 4 - belt_height / 2,
            self.HEIGHT / 2 - belt_height / 2,
            self.HEIGHT * 3 / 4 - belt_height / 2
        ]
        
        # Draw matching zone first (underneath belts)
        zone_rect = pygame.Rect(self.BLOCK_SPACING/2 - self.BLOCK_SIZE/2 - 5, 0, self.BLOCK_SIZE + 10, self.HEIGHT)
        zone_surface = pygame.Surface(zone_rect.size, pygame.SRCALPHA)
        zone_surface.fill(self.COLOR_ZONE)
        self.screen.blit(zone_surface, zone_rect.topleft)
        pygame.gfxdraw.rectangle(self.screen, zone_rect, (*self.COLOR_ZONE[:3], 150))

        # Draw belts
        for i, y_pos in enumerate(belt_y_positions):
            belt_rect = pygame.Rect(0, y_pos, self.WIDTH, belt_height)
            pygame.gfxdraw.rectangle(self.screen, belt_rect, self.COLOR_BELT)

            # Draw rotation feedback glow
            for r_belt_idx, r_alpha in self.rotation_feedback:
                if r_belt_idx == i:
                    glow_color = (*self.COLOR_SCORE, r_alpha)
                    glow_surf = pygame.Surface(belt_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=5)
                    self.screen.blit(glow_surf, belt_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)


        # --- Blocks ---
        for i in range(self.NUM_BELTS):
            belt_y = belt_y_positions[i] + belt_height / 2
            if i < len(self.belts) and i < len(self.block_positions):
                for j in range(len(self.belts[i])):
                    if j < len(self.block_positions[i]):
                        color_idx = self.belts[i][j]
                        block_color = self.BLOCK_COLORS[color_idx]
                        block_x = self.block_positions[i][j]
                        
                        if 0 < block_x < self.WIDTH:
                            rect = pygame.Rect(
                                int(block_x - self.BLOCK_SIZE / 2),
                                int(belt_y - self.BLOCK_SIZE / 2),
                                self.BLOCK_SIZE, self.BLOCK_SIZE
                            )
                            pygame.draw.rect(self.screen, block_color, rect, border_radius=5)
                            pygame.draw.rect(self.screen, tuple(c*0.7 for c in block_color), rect, width=3, border_radius=5)

        # --- Particles ---
        for p in self.particles:
            x, y, _, _, lifetime, color = p
            alpha = max(0, min(255, int(255 * (lifetime / 40.0))))
            radius = int(self.BLOCK_SIZE / 8 * (lifetime / 40.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), radius, (*color, alpha))

        # --- UI ---
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (20, 10))

        highscore_text = self.font_small.render(f"HIGH: {GameEnv.high_score}", True, self.COLOR_TEXT)
        self.screen.blit(highscore_text, (20, 50))
        
        steps_text = self.font_small.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 20, 10))


    def close(self):
        pygame.quit()
        
# --- Example Usage ---
if __name__ == "__main__":
    # This part is for interactive testing and visualization.
    # It will not run in the headless evaluation environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a display surface
    pygame.display.set_caption("Conveyor Belt Puzzle")
    main_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    running = True
    total_reward = 0
    
    # Manual control action
    current_action = [0, 0, 0] # No-op
    
    while running:
        # --- Event Handling for Manual Control ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    current_action[0] = 1 # Top belt
                elif event.key == pygame.K_DOWN:
                    current_action[0] = 2 # Bottom belt
                elif event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    current_action[0] = 3 # Middle belt
                elif event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
                elif event.key == pygame.K_ESCAPE:
                    running = False
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    current_action[0] = 0 # No-op

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(current_action)
        total_reward += reward
        
        # --- Rendering ---
        # The environment renders the state to its internal surface.
        # We get that surface and display it.
        # The observation `obs` is a numpy array, but for direct display,
        # it's easier to use the pygame surface directly.
        main_screen.blit(env.screen, (0, 0))
        pygame.display.flip()

        # Reset action to no-op for step-by-step control if not holding a key
        # For a more responsive feel, you might handle this differently,
        # but for this example, we send one action per frame.
        # current_action = [0, 0, 0] 

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    env.close()