import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:05:30.425505
# Source Brief: brief_00806.md
# Brief Index: 806
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An alchemy lab environment where the agent mixes reagents to create a potent potion.

    The goal is to discover reagent combinations that increase a potion's potency
    without causing the lab's overall instability to reach a critical level,
    which would result in an explosion.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) to cycle selection.
    - action[1]: Space button (0=released, 1=held) to select a source container.
    - action[2]: Shift button (0=released, 1=held) to select a target container and trigger a reaction.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 per point of potency gained in a reaction.
    - -0.1 per point of instability gained in a reaction.
    - +1 for successfully completing a reaction.
    - +100 for winning (potency >= 100).
    - -10 for losing (instability >= 100).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Mix reagents in an alchemy lab to create a potent potion. Discover powerful combinations "
        "but be careful not to let the lab's instability cause a catastrophic explosion."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to select containers. Press space to select a source "
        "reagent, then press shift on a target flask to mix them."
    )
    auto_advance = False

    # --- CONSTANTS ---
    # Sizing
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    WIN_POTENCY = 100
    MAX_INSTABILITY = 100

    # Colors
    COLOR_BG = (14, 18, 25) # Dark blue-grey
    COLOR_BENCH = (35, 43, 58)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_POTENCY = (137, 220, 255)
    COLOR_INSTABILITY_GOOD = (60, 255, 150)
    COLOR_INSTABILITY_MID = (255, 220, 100)
    COLOR_INSTABILITY_BAD = (255, 80, 80)
    COLOR_SELECT_GLOW = (255, 255, 255)
    COLOR_SOURCE_GLOW = (100, 255, 100)
    COLOR_TARGET_GLOW = (100, 100, 255)

    # Physics & Animation
    PARTICLE_LIFESPAN = 60
    PARTICLE_SPEED = 2.5
    BUBBLE_LIFESPAN = 45
    EXPLOSION_PARTICLES = 200

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        self.render_mode = render_mode

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_instability = 0.0
        
        self.dispensers = []
        self.flasks = []
        
        self.selectable_items = []
        self.selection_index = 0
        self.source_selection = None
        self.target_selection = None
        
        self.particles = []
        self.last_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Core State ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_instability = 0.0
        self.last_reward = 0.0

        # --- Reset Selections & Animations ---
        self.selection_index = 0
        self.source_selection = None
        self.target_selection = None
        self.particles = []
        
        # --- Procedurally Generate Reagents ---
        self.dispensers = self._generate_dispensers(4)
        
        # --- Initialize Flasks ---
        self.flasks = [
            {"pos": (self.WIDTH * 0.25, 320), "contents": [], "potency": 0.0, "stability": 100.0, "bubbles": []}
            for i in range(3)
        ]
        self.flasks[0]['pos'] = (self.WIDTH * 0.25, 320)
        self.flasks[1]['pos'] = (self.WIDTH * 0.50, 320)
        self.flasks[2]['pos'] = (self.WIDTH * 0.75, 320)

        # --- Define Selectable Items and their Grid ---
        # Layout: 4 dispensers on top, 3 flasks on bottom
        self.selectable_items = [
            {"id": f"D{i}", "pos": d["pos"], "type": "dispenser"} for i, d in enumerate(self.dispensers)
        ] + [
            {"id": f"F{i}", "pos": f["pos"], "type": "flask"} for i, f in enumerate(self.flasks)
        ]
        
        # Navigation map for up/down/left/right actions
        # Maps from_index to to_index for each direction
        self.nav_map = {
            # Dispensers (0-3)
            0: {"right": 1, "down": 4}, 
            1: {"left": 0, "right": 2, "down": 5},
            2: {"left": 1, "right": 3, "down": 5},
            3: {"left": 2, "down": 6},
            # Flasks (4-6)
            4: {"up": 0, "right": 5},
            5: {"up": 1, "left": 4, "right": 6},
            6: {"up": 3, "left": 5},
        }

        return self._get_observation(), self._get_info()

    def _generate_dispensers(self, count):
        all_colors = [
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (255, 128, 0), (128, 0, 255), (0, 255, 128)
        ]
        self.np_random.shuffle(all_colors)
        colors = all_colors[:count]
        
        dispensers = []
        base_volatility = 1.0 + (self.score // 20) * 0.2
        
        for i in range(count):
            dispensers.append({
                "id": self.np_random.integers(1000, 10000),
                "pos": (self.WIDTH * (i + 1) / (count + 1), 100),
                "color": colors[i],
                "volatility": self.np_random.uniform(base_volatility * 0.5, base_volatility * 1.5),
                "potency_mod": self.np_random.uniform(0.8, 1.2),
                "stability_mod": self.np_random.uniform(0.8, 1.2)
            })
        return dispensers

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        # --- 1. Handle Movement/Selection ---
        if movement > 0:
            direction_map = {1: "up", 2: "down", 3: "left", 4: "right"}
            direction = direction_map[movement]
            if direction in self.nav_map.get(self.selection_index, {}):
                self.selection_index = self.nav_map[self.selection_index][direction]

        # --- 2. Handle Source/Target Setting ---
        selected_item = self.selectable_items[self.selection_index]
        if space_pressed:
            if selected_item["type"] == "dispenser":
                self.source_selection = self.selection_index
                # SFX: Source select chime
        
        if shift_pressed:
            if selected_item["type"] == "flask":
                self.target_selection = self.selection_index
                # SFX: Target select chime

        # --- 3. Trigger Reaction if Valid Pair is Selected ---
        if self.source_selection is not None and self.target_selection is not None:
            source_item = self.selectable_items[self.source_selection]
            target_item = self.selectable_items[self.target_selection]

            if source_item["type"] == "dispenser" and target_item["type"] == "flask":
                source_idx = int(source_item["id"][1:])
                target_idx = int(target_item["id"][1:])
                
                reaction_reward = self._perform_reaction(source_idx, target_idx)
                reward += reaction_reward
                self.last_reward = reaction_reward
            
            # Reset selections after an attempt
            self.source_selection = None
            self.target_selection = None

        # --- 4. Update Game State & Check Termination ---
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if any(f['potency'] >= self.WIN_POTENCY for f in self.flasks):
                reward += 100 # Win bonus
                # SFX: Victory fanfare
            elif self.total_instability >= self.MAX_INSTABILITY:
                reward -= 10 # Loss penalty
                self._create_explosion()
                # SFX: Large explosion
            # No reward change for max steps termination
        
        self.score += reward
        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _perform_reaction(self, dispenser_idx, flask_idx):
        reagent = self.dispensers[dispenser_idx]
        flask = self.flasks[flask_idx]
        
        # --- Create Particle Stream Animation ---
        start_pos = reagent['pos']
        end_pos = flask['pos']
        for _ in range(20):
            self.particles.append(self._create_stream_particle(start_pos, end_pos, reagent['color']))

        # --- Calculate Reaction based on a deterministic "secret" formula ---
        # The hash ensures the same combination always yields the same result
        all_reagent_ids = tuple(sorted([r['id'] for r in flask['contents']] + [reagent['id']]))
        reaction_seed = hash(all_reagent_ids)
        rng = random.Random(reaction_seed)

        potency_gain = rng.uniform(2, 10) * reagent['potency_mod']
        if len(flask['contents']) > 0:
            potency_gain *= rng.choice([0.5, 1.0, 1.5, 2.0]) # Synergies/Antagonisms

        instability_gain = rng.uniform(1, 5) * reagent['volatility']
        
        # --- Update Flask and Global State ---
        old_potency = flask['potency']
        flask['contents'].append(reagent)
        flask['potency'] = max(0, flask['potency'] + potency_gain)
        self.total_instability = min(self.MAX_INSTABILITY, self.total_instability + instability_gain)
        
        # Add bubbles for visual feedback
        for _ in range(int(potency_gain + instability_gain)):
            flask['bubbles'].append(self._create_bubble_particle(flask['pos']))
        # SFX: Potion bubbling sound

        # --- Calculate Reward for this action ---
        reward = 1.0 # Base reward for successful reaction
        reward += (flask['potency'] - old_potency) * 0.1
        reward -= instability_gain * 0.1
        
        return reward

    def _check_termination(self):
        win = any(f['potency'] >= self.WIN_POTENCY for f in self.flasks)
        loss_explosion = self.total_instability >= self.MAX_INSTABILITY
        loss_timeout = self.steps >= self.MAX_STEPS
        return win or loss_explosion or loss_timeout

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
            "instability": self.total_instability,
            "max_potency": max(f['potency'] for f in self.flasks) if self.flasks else 0,
        }

    def close(self):
        pygame.quit()

    # --- Rendering Methods ---
    def _render_game(self):
        # Draw workbench
        pygame.draw.rect(self.screen, self.COLOR_BENCH, (0, self.HEIGHT - 120, self.WIDTH, 120))
        pygame.draw.line(self.screen, (50, 60, 80), (0, self.HEIGHT-120), (self.WIDTH, self.HEIGHT-120), 3)

        # Draw dispensers
        for i, d in enumerate(self.dispensers):
            self._draw_dispenser(d, i)

        # Draw flasks
        for i, f in enumerate(self.flasks):
            self._draw_flask(f, i)

        # Draw particles
        for p in self.particles:
            self._draw_particle(p)

    def _draw_dispenser(self, dispenser, index):
        pos = dispenser['pos']
        size = 20
        rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
        pygame.draw.rect(self.screen, dispenser['color'], rect, border_radius=5)
        pygame.draw.rect(self.screen, tuple(c*0.7 for c in dispenser['color']), rect, width=3, border_radius=5)
        self._draw_selection_glow(index, rect.center, size + 5)

    def _draw_flask(self, flask, index):
        pos = flask['pos']
        w, h = 60, 80
        
        # Draw flask contents (layered colors)
        if flask['contents']:
            content_h = (h - 10) * (min(len(flask['contents']), 10) / 10.0)
            layer_h = content_h / len(flask['contents'])
            for i, reagent in enumerate(flask['contents']):
                pygame.draw.rect(self.screen, reagent['color'], (pos[0] - w/2 + 5, pos[1] + h/2 - (i+1)*layer_h, w - 10, layer_h))

        # Draw flask outline (as a series of lines for a beaker shape)
        p = [
            (pos[0] - w/2, pos[1] - h/2), (pos[0] + w/2, pos[1] - h/2),
            (pos[0] + w/2, pos[1] + h/2 - 10), (pos[0] + w/2 - 10, pos[1] + h/2),
            (pos[0] - w/2 + 10, pos[1] + h/2), (pos[0] - w/2, pos[1] + h/2 - 10)
        ]
        pygame.draw.lines(self.screen, self.COLOR_UI_TEXT, True, p, 3)
        
        # Draw bubbles
        for b in flask['bubbles']:
            self._draw_particle(b)

        # Draw potency text
        potency_text = self.font_small.render(f"P: {flask['potency']:.1f}", True, self.COLOR_POTENCY)
        self.screen.blit(potency_text, (pos[0] - potency_text.get_width() / 2, pos[1] - h/2 - 20))
        
        self._draw_selection_glow(index + 4, pos, h / 2 + 5)

    def _draw_selection_glow(self, item_index, pos, radius):
        if self.game_over: return
        
        color = None
        if self.source_selection == item_index:
            color = self.COLOR_SOURCE_GLOW
        elif self.target_selection == item_index:
            color = self.COLOR_TARGET_GLOW
        elif self.selection_index == item_index:
            color = self.COLOR_SELECT_GLOW
        
        if color:
            for i in range(5):
                alpha = 150 - i * 30
                if alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), int(radius + i), (*color, alpha))

    def _render_ui(self):
        # --- Instability Bar ---
        bar_width = self.WIDTH - 40
        instability_ratio = self.total_instability / self.MAX_INSTABILITY
        current_width = bar_width * instability_ratio
        
        # Interpolate color
        if instability_ratio < 0.5:
            c = self.COLOR_INSTABILITY_GOOD
        elif instability_ratio < 0.8:
            c = self.COLOR_INSTABILITY_MID
        else:
            c = self.COLOR_INSTABILITY_BAD
            
        pygame.draw.rect(self.screen, (50, 50, 50), (20, 20, bar_width, 20), border_radius=5)
        if current_width > 0:
            pygame.draw.rect(self.screen, c, (20, 20, current_width, 20), border_radius=5)
        inst_text = self.font_main.render(f"SYSTEM INSTABILITY: {self.total_instability:.1f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(inst_text, (25, 19))

        # --- Score and Step Count ---
        score_text = self.font_main.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, self.HEIGHT - 30))
        
        step_text = self.font_main.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(step_text, (self.WIDTH - step_text.get_width() - 20, self.HEIGHT - 30))

        # --- Last Reward ---
        reward_color = (100, 255, 100) if self.last_reward >= 0 else (255, 100, 100)
        reward_text = self.font_small.render(f"Last Reward: {self.last_reward:+.2f}", True, reward_color)
        self.screen.blit(reward_text, (self.WIDTH - reward_text.get_width() - 20, 20))

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            if any(f['potency'] >= self.WIN_POTENCY for f in self.flasks):
                msg = "POTION SYNTHESIS COMPLETE"
                color = self.COLOR_POTENCY
            else:
                msg = "CATASTROPHIC FAILURE"
                color = self.COLOR_INSTABILITY_BAD
            
            end_text = self.font_main.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))


    # --- Particle System ---
    def _create_stream_particle(self, start_pos, end_pos, color):
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        velocity = (math.cos(angle) * self.PARTICLE_SPEED, math.sin(angle) * self.PARTICLE_SPEED)
        return {
            "pos": list(start_pos), "vel": velocity, "lifespan": self.PARTICLE_LIFESPAN,
            "type": "stream", "color": color, "max_life": self.PARTICLE_LIFESPAN
        }

    def _create_bubble_particle(self, flask_pos):
        pos = [flask_pos[0] + self.np_random.uniform(-15, 15), flask_pos[1] + 30]
        vel = [self.np_random.uniform(-0.3, 0.3), self.np_random.uniform(-0.8, -0.2)]
        return {
            "pos": pos, "vel": vel, "lifespan": self.BUBBLE_LIFESPAN,
            "type": "bubble", "color": (200, 220, 255), "max_life": self.BUBBLE_LIFESPAN
        }

    def _create_explosion_particle(self):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(2, 8)
        vel = (math.cos(angle) * speed, math.sin(angle) * speed)
        color_choices = [self.COLOR_INSTABILITY_BAD, self.COLOR_INSTABILITY_MID, (255, 255, 255)]
        color = color_choices[self.np_random.integers(len(color_choices))]
        return {
            "pos": [self.WIDTH / 2, self.HEIGHT / 2], "vel": vel, "lifespan": 60,
            "type": "explosion", "color": color, "max_life": 60
        }

    def _create_explosion(self):
        for _ in range(self.EXPLOSION_PARTICLES):
            self.particles.append(self._create_explosion_particle())
    
    def _update_particles(self):
        # Update stream and explosion particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['type'] == 'explosion':
                p['vel'] = (p['vel'][0] * 0.98, p['vel'][1] * 0.98) # Friction

        # Update bubbles in flasks
        for f in self.flasks:
            f['bubbles'] = [b for b in f['bubbles'] if b['lifespan'] > 0]
            for b in f['bubbles']:
                b['pos'][0] += b['vel'][0]
                b['pos'][1] += b['vel'][1]
                b['lifespan'] -= 1

    def _draw_particle(self, p):
        life_ratio = p['lifespan'] / p['max_life']
        pos = (int(p['pos'][0]), int(p['pos'][1]))
        
        if p['type'] == 'stream':
            size = int(3 * life_ratio)
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)
        elif p['type'] == 'bubble':
            alpha = int(150 * life_ratio)
            if alpha > 0:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 2, (*p['color'], alpha))
        elif p['type'] == 'explosion':
            size = int(8 * life_ratio)
            if size > 0:
                color_with_alpha = (*p['color'], int(255 * life_ratio))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color_with_alpha)

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To display the game, we need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Controls Mapping ---
    # Arrows: Move selection
    # Space: Select source
    # Shift: Select target (triggers action)
    # R: Reset environment
    
    action = [0, 0, 0] # [movement, space, shift]
    
    running = True
    display_surf = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    # Initial render
    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    display_surf.blit(surf, (0, 0))
    pygame.display.flip()

    while running:
        # Pygame event handling
        event_happened = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                event_happened = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    action = [0, 0, 0]
                elif event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
        
        # Step the environment only if an action was taken
        if event_happened:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset action for next frame (important for key presses)
            action = [0, 0, 0]
            
            # Render the observation to the screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_surf.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Episode Finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
                # Wait a bit before auto-resetting
                pygame.time.wait(2000)
                obs, info = env.reset()
                
                # Render the reset state
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                display_surf.blit(surf, (0, 0))
                pygame.display.flip()

        env.clock.tick(60) # Limit loop speed

    env.close()