import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:11:59.692144
# Source Brief: brief_00274.md
# Brief Index: 274
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment simulating protein synthesis by a ribosome.

    The agent controls a ribosome moving along an mRNA strand. The goal is to
    synthesize a protein of 100 amino acids by pressing the 'synthesize' button.
    Each synthesis action adds 5 amino acids but has a chance to introduce an error,
    which reduces the protein's stability.

    - Victory: Synthesize a 100-amino-acid protein.
    - Failure: Protein stability drops to zero.
    - Action Space: MultiDiscrete([5, 2, 2]) for movement, synthesis, and an unused action.
    - Observation Space: 640x400 RGB image of the game state.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a ribosome moving along an mRNA strand to synthesize a protein. "
        "Each synthesis action adds to the protein but risks introducing an error that reduces stability."
    )
    user_guide = "Controls: Use ←→ arrow keys to move the ribosome. Press space to synthesize the protein."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors (Bright/Saturated for interactive, Dark/Desaturated for background)
    COLOR_BG = (10, 20, 35)
    COLOR_BG_PARTICLE = (30, 40, 65)
    COLOR_MRNA = (40, 60, 90)
    COLOR_MRNA_CODON = (60, 80, 110)
    COLOR_RIBOSOME = (0, 150, 255)
    COLOR_RIBOSOME_GLOW = (0, 100, 200)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SUCCESS = (0, 255, 100)
    COLOR_ERROR = (255, 50, 50)
    COLOR_STABILITY_HIGH = (0, 220, 120)
    COLOR_STABILITY_LOW = (220, 50, 50)
    
    # Game Parameters
    MAX_STEPS = 1000
    WIN_LENGTH = 100
    RIBOSOME_SPEED = 8
    SYNTHESIS_AMOUNT = 5
    ERROR_CHANCE = 0.20
    STABILITY_LOSS_PER_ERROR = 5.0
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # --- State Variables ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.ribosome_x = None
        self.protein_length = None
        self.stability = None
        self.error_count = None
        self.last_space_held = None
        self.particles = None
        self.bg_particles = None
        
        # self.reset() is called by the wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Initialize Game State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.ribosome_x = self.SCREEN_WIDTH // 2
        self.protein_length = 0
        self.stability = 100.0
        self.error_count = 0
        
        self.last_space_held = False
        self.particles = []
        
        # Create a persistent field of background particles
        if not hasattr(self, 'bg_particles') or self.bg_particles is None:
            self.bg_particles = [
                {
                    "pos": [self.np_random.uniform(0, self.SCREEN_WIDTH), self.np_random.uniform(0, self.SCREEN_HEIGHT)],
                    "vel": [self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2)],
                    "radius": self.np_random.uniform(1, 3)
                } for _ in range(50)
            ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- Unpack Actions ---
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused in this design

        # --- Handle Movement ---
        if movement == 3:  # Left
            self.ribosome_x -= self.RIBOSOME_SPEED
        elif movement == 4: # Right
            self.ribosome_x += self.RIBOSOME_SPEED
        
        # Clamp ribosome position to be within mRNA bounds
        self.ribosome_x = np.clip(self.ribosome_x, 20, self.SCREEN_WIDTH - 20)

        # --- Handle Synthesis ---
        # Detect rising edge of spacebar press
        if space_held and not self.last_space_held:
            # SFX: Synthesize_start.wav
            synthesis_successes = 0
            synthesis_errors = 0
            for _ in range(self.SYNTHESIS_AMOUNT):
                if self.protein_length < self.WIN_LENGTH:
                    self.protein_length += 1
                    
                    if self.np_random.random() < self.ERROR_CHANCE:
                        # Error occurred
                        synthesis_errors += 1
                        self.error_count += 1
                        self.stability = max(0, self.stability - self.STABILITY_LOSS_PER_ERROR)
                        reward -= 1.0
                        # SFX: Error.wav
                        self._create_particles(self.COLOR_ERROR, 15)
                    else:
                        # Successful synthesis
                        synthesis_successes += 1
                        reward += 0.1
                        # SFX: Success_ding.wav
                        self._create_particles(self.COLOR_SUCCESS, 5, speed_mult=0.5)
        
        self.last_space_held = space_held

        # --- Update Game Logic ---
        self._update_particles()
        
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.protein_length >= self.WIN_LENGTH:
                reward += 100  # Victory bonus
                # SFX: Victory_fanfare.wav
            elif self.stability <= 0:
                reward -= 100  # Failure penalty
                # SFX: Failure_buzz.wav
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _check_termination(self):
        if self.protein_length >= self.WIN_LENGTH:
            self.game_over = True
        elif self.stability <= 0:
            self.game_over = True
        
        return self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "protein_length": self.protein_length,
            "stability": self.stability,
            "error_count": self.error_count,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_particles()
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array in the required format
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, 3), we want (height, width, 3)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background_particles(self):
        if not hasattr(self, 'bg_particles') or self.bg_particles is None:
            return
        for p in self.bg_particles:
            p["pos"][0] = (p["pos"][0] + p["vel"][0]) % self.SCREEN_WIDTH
            p["pos"][1] = (p["pos"][1] + p["vel"][1]) % self.SCREEN_HEIGHT
            pygame.gfxdraw.aacircle(
                self.screen, int(p["pos"][0]), int(p["pos"][1]), int(p["radius"]), self.COLOR_BG_PARTICLE
            )

    def _render_game(self):
        # --- Draw mRNA Strand ---
        mrna_y = self.SCREEN_HEIGHT // 2
        pygame.draw.line(self.screen, self.COLOR_MRNA, (0, mrna_y), (self.SCREEN_WIDTH, mrna_y), 4)
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_MRNA_CODON, (x, mrna_y - 5), (x, mrna_y + 5), 1)

        # --- Draw Synthesized Protein Bar ---
        protein_bar_y = mrna_y - 40
        protein_bar_height = 20
        max_width = self.SCREEN_WIDTH - 40
        current_width = int(max_width * (self.protein_length / self.WIN_LENGTH))

        # Stability color gradient
        stability_percent = self.stability / 100.0
        protein_color = (
            int(self.COLOR_STABILITY_LOW[0] * (1 - stability_percent) + self.COLOR_STABILITY_HIGH[0] * stability_percent),
            int(self.COLOR_STABILITY_LOW[1] * (1 - stability_percent) + self.COLOR_STABILITY_HIGH[1] * stability_percent),
            int(self.COLOR_STABILITY_LOW[2] * (1 - stability_percent) + self.COLOR_STABILITY_HIGH[2] * stability_percent),
        )

        if current_width > 0:
            pygame.draw.rect(
                self.screen, protein_color, (20, protein_bar_y, current_width, protein_bar_height),
                border_radius=5
            )
        pygame.draw.rect(
            self.screen, self.COLOR_MRNA, (20, protein_bar_y, max_width, protein_bar_height), 2,
            border_radius=5
        )

        # --- Draw Particles ---
        if self.particles:
            for p in self.particles:
                pos = p["pos"]
                radius = p["radius"] * (p["life"] / p["max_life"])
                if radius > 1:
                    pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), int(radius), p["color"])

        # --- Draw Ribosome ---
        ribosome_y = mrna_y
        ribosome_size = 18

        # Glow effect
        for i in range(10, 0, -2):
            alpha = 80 * (1 - i / 10)
            glow_color = (*self.COLOR_RIBOSOME_GLOW, alpha)
            s = pygame.Surface((ribosome_size*2 + i*2, ribosome_size*2 + i*2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (s.get_width()//2, s.get_height()//2), ribosome_size + i)
            self.screen.blit(s, (self.ribosome_x - s.get_width()//2, ribosome_y - s.get_height()//2))

        # Main body
        pygame.gfxdraw.filled_circle(self.screen, int(self.ribosome_x), int(ribosome_y), ribosome_size, self.COLOR_RIBOSOME)
        pygame.gfxdraw.aacircle(self.screen, int(self.ribosome_x), int(ribosome_y), ribosome_size, self.COLOR_TEXT)

    def _render_ui(self):
        # --- Info Text ---
        texts = [
            f"Protein Length: {self.protein_length}/{self.WIN_LENGTH}",
            f"Stability: {self.stability:.1f}%",
            f"Errors: {self.error_count}",
        ]
        for i, text in enumerate(texts):
            text_surf = self.font_main.render(text, True, self.COLOR_TEXT)
            self.screen.blit(text_surf, (15, 15 + i * 25))
            
        # --- Score Text ---
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 15, 15))

        # --- Game Over Message ---
        if self.game_over:
            if self.protein_length >= self.WIN_LENGTH:
                msg = "SYNTHESIS COMPLETE"
                color = self.COLOR_SUCCESS
            else:
                msg = "PROTEIN UNSTABLE"
                color = self.COLOR_ERROR
            
            msg_surf = self.font_big.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 60))
            
            # Add a dark background for readability
            bg_rect = msg_rect.inflate(20, 10)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 150))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": [self.ribosome_x, self.SCREEN_HEIGHT // 2],
                "vel": vel,
                "color": color,
                "radius": self.np_random.uniform(3, 8),
                "life": self.np_random.uniform(20, 40),
                "max_life": 40
            })

    def _update_particles(self):
        if not self.particles:
            return
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95  # Drag
            p["vel"][1] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # This part is not needed for the agent but is useful for development
    
    # Check if a display is available, otherwise skip human play
    try:
        pygame.display.init()
        pygame.font.init()
    except pygame.error:
        print("No display available, skipping human play test.")
        # Create a dummy env to test basic API calls
        env = GameEnv()
        print("Testing reset...")
        obs, info = env.reset()
        print("...reset OK")
        print("Testing step...")
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print("...step OK")
        env.close()
        exit()

    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Ribosome Runner")
    clock = pygame.time.Clock()
    running = True

    while running:
        # --- Action mapping from keyboard ---
        movement = 0 # none
        space = 0
        shift = 0 # unused
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space = 1

        action = [movement, space, shift]
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R'
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Environment step ---
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        frame = np.transpose(obs, (1, 0, 2)) # Transpose back to (width, height, 3) for pygame
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()