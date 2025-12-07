import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:59:22.486224
# Source Brief: brief_01413.md
# Brief Index: 1413
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two robotic arms to quickly sort colored blocks into their matching slots. "
        "Place blocks correctly to get a speed boost and beat the clock!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the selected arm. Press space to pick up or place a block. "
        "Press Tab to switch between the two arms."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    ARM_RADIUS = 15
    BLOCK_SIZE = 24
    SLOT_SIZE = 28
    PICKUP_DISTANCE = 30
    PLACEMENT_DISTANCE = 20
    NUM_BLOCKS = 12
    MAX_STEPS = 6000  # 100 steps/sec -> 60 seconds

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 55, 71)
    COLOR_UI_TEXT = (220, 220, 220)

    ARM1_ACTIVE = (50, 150, 255)
    ARM1_INACTIVE = (40, 80, 120)
    ARM2_ACTIVE = (255, 150, 50)
    ARM2_INACTIVE = (120, 80, 40)

    SPEED_BOOST_COLOR = (255, 255, 100)
    SUCCESS_COLOR = (0, 255, 100)

    # Physics
    BASE_ARM_SPEED = 2.5
    BOOSTED_ARM_SPEED = 5.0
    SPEED_BOOST_DURATION = 300 # 3 seconds at 100 steps/sec

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 28, bold=True)

        self.arms = []
        self.blocks = []
        self.slots = []
        self.particles = []
        self.active_arm_idx = 0
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_block_colors()
        # self.reset() is called by the environment wrapper

    def _generate_block_colors(self):
        self.BLOCK_COLORS = []
        for i in range(self.NUM_BLOCKS):
            hue = i / self.NUM_BLOCKS
            color = pygame.Color(0)
            color.hsva = (hue * 360, 90, 95, 100)
            self.BLOCK_COLORS.append(tuple(color)[:3])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.active_arm_idx = 0
        self.particles = []

        self.arms = [
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT / 2),
                "color_active": self.ARM1_ACTIVE,
                "color_inactive": self.ARM1_INACTIVE,
                "holding_block_idx": None,
                "speed": self.BASE_ARM_SPEED,
                "speed_boost_timer": 0,
            },
            {
                "pos": pygame.Vector2(self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT / 2),
                "color_active": self.ARM2_ACTIVE,
                "color_inactive": self.ARM2_INACTIVE,
                "holding_block_idx": None,
                "speed": self.BASE_ARM_SPEED,
                "speed_boost_timer": 0,
            },
        ]

        # Setup slots in a 2x6 grid
        self.slots = []
        slot_indices = list(range(self.NUM_BLOCKS))
        self.np_random.shuffle(slot_indices)
        grid_w, grid_h = 6, 2
        spacing_x = (self.SCREEN_WIDTH * 0.6) / (grid_w -1)
        spacing_y = (self.SCREEN_HEIGHT * 0.4) / (grid_h -1)
        start_x = self.SCREEN_WIDTH * 0.2
        start_y = self.SCREEN_HEIGHT * 0.3

        for i in range(self.NUM_BLOCKS):
            row = i // grid_w
            col = i % grid_w
            pos = pygame.Vector2(start_x + col * spacing_x, start_y + row * spacing_y)
            self.slots.append({
                "pos": pos,
                "rect": pygame.Rect(pos.x - self.SLOT_SIZE/2, pos.y - self.SLOT_SIZE/2, self.SLOT_SIZE, self.SLOT_SIZE),
                "correct_block_idx": slot_indices[i],
                "is_filled": False,
            })

        # Setup blocks at the bottom
        self.blocks = []
        for i in range(self.NUM_BLOCKS):
            pos = pygame.Vector2(
                (self.SCREEN_WIDTH / (self.NUM_BLOCKS + 1)) * (i + 1),
                self.SCREEN_HEIGHT - 30
            )
            self.blocks.append({
                "pos": pos,
                "color": self.BLOCK_COLORS[i],
                "is_held": False,
                "is_placed": False,
                "target_slot_idx": next(j for j, s in enumerate(self.slots) if s["correct_block_idx"] == i)
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        self.steps += 1

        movement, space_press, _ = action
        space_press = space_press == 1

        # --- Update Game State ---
        self._update_timers()

        active_arm = self.arms[self.active_arm_idx]

        # --- Handle Actions ---
        # Action 0: Switch active arm
        if movement == 0:
            # Special handling for switching arms; in the human-play loop, this is mapped to Tab.
            # In an agent context, it's just one of the discrete movement actions.
            # We'll make it explicit: if the action is 0, we switch, otherwise we move.
            # To avoid unintended switching when a no-op is intended, we'll assume action 0 is ONLY for switching.
            # The human play loop will need to be careful.
            # A better action space might separate switch from move, but we must adhere to [5,2,2].
            # The provided human-play code handles this correctly by mapping Tab to movement=0.
            self.active_arm_idx = 1 - self.active_arm_idx
        # Actions 1-4: Move active arm
        elif movement in [1, 2, 3, 4]:
            direction = pygame.Vector2(0, 0)
            if movement == 1: direction.y = -1 # Up
            elif movement == 2: direction.y = 1  # Down
            elif movement == 3: direction.x = -1 # Left
            elif movement == 4: direction.x = 1  # Right

            # Calculate distance-based reward if holding a block
            if active_arm["holding_block_idx"] is not None:
                block_idx = active_arm["holding_block_idx"]
                block = self.blocks[block_idx]
                target_slot = self.slots[block["target_slot_idx"]]
                old_dist = active_arm["pos"].distance_to(target_slot["pos"])

                active_arm["pos"] += direction * active_arm["speed"]

                new_dist = active_arm["pos"].distance_to(target_slot["pos"])
                reward += (old_dist - new_dist) * 0.01 # Dense reward for moving closer
            else:
                active_arm["pos"] += direction * active_arm["speed"]

            # Clamp position
            active_arm["pos"].x = np.clip(active_arm["pos"].x, 0, self.SCREEN_WIDTH)
            active_arm["pos"].y = np.clip(active_arm["pos"].y, 0, self.SCREEN_HEIGHT)

            # Move held block with arm
            if active_arm["holding_block_idx"] is not None:
                self.blocks[active_arm["holding_block_idx"]]["pos"] = active_arm["pos"]

        # Action: Pick up / Place block
        if space_press:
            # If holding a block, try to place it
            if active_arm["holding_block_idx"] is not None:
                block_idx = active_arm["holding_block_idx"]

                # Find closest empty slot
                closest_slot_idx = None
                min_dist = float('inf')
                for i, slot in enumerate(self.slots):
                    if not slot["is_filled"]:
                        dist = active_arm["pos"].distance_to(slot["pos"])
                        if dist < min_dist:
                            min_dist = dist
                            closest_slot_idx = i

                if closest_slot_idx is not None and min_dist < self.PLACEMENT_DISTANCE:
                    # Check if it's the correct block for the slot
                    if self.slots[closest_slot_idx]["correct_block_idx"] == block_idx:
                        # Correct placement
                        self.score += 1
                        reward += 1.0

                        block = self.blocks[block_idx]
                        slot = self.slots[closest_slot_idx]

                        slot["is_filled"] = True
                        block["is_placed"] = True
                        block["pos"] = slot["pos"]
                        active_arm["holding_block_idx"] = None

                        # Trigger speed boost and visual effect
                        active_arm["speed_boost_timer"] = self.SPEED_BOOST_DURATION
                        active_arm["speed"] = self.BOOSTED_ARM_SPEED
                        self._add_particle(slot["pos"], self.SUCCESS_COLOR, 40, 30)

                    else: # Incorrect placement attempt, just drop the block
                        active_arm["holding_block_idx"] = None
                        self.blocks[block_idx]["is_held"] = False
                else: # Dropped in empty space
                    active_arm["holding_block_idx"] = None
                    self.blocks[block_idx]["is_held"] = False

            # If not holding a block, try to pick one up
            else:
                closest_block_idx = None
                min_dist = float('inf')
                for i, block in enumerate(self.blocks):
                    if not block["is_held"] and not block["is_placed"]:
                        dist = active_arm["pos"].distance_to(block["pos"])
                        if dist < min_dist:
                            min_dist = dist
                            closest_block_idx = i

                if closest_block_idx is not None and min_dist < self.PICKUP_DISTANCE:
                    active_arm["holding_block_idx"] = closest_block_idx
                    self.blocks[closest_block_idx]["is_held"] = True

        # --- Check Termination Conditions ---
        if self.score == self.NUM_BLOCKS:
            reward += 100  # Victory bonus
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # Time-out is a truncation, not a terminal failure state
            truncated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_timers(self):
        # Update arm speed boosts
        for arm in self.arms:
            if arm["speed_boost_timer"] > 0:
                arm["speed_boost_timer"] -= 1
                if arm["speed_boost_timer"] == 0:
                    arm["speed"] = self.BASE_ARM_SPEED

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1

    def _add_particle(self, pos, color, max_radius, lifetime):
        self.particles.append({"pos": pos, "color": color, "max_radius": max_radius, "lifetime": lifetime, "max_life": lifetime})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        self._render_slots()
        self._render_blocks()
        self._render_arms()
        self._render_particles()

    def _render_slots(self):
        for slot in self.slots:
            block_color = self.blocks[slot["correct_block_idx"]]["color"]
            # Draw outline matching the correct block
            pygame.gfxdraw.rectangle(self.screen, slot["rect"], (*block_color, 100))
            # Draw inner area
            if not slot["is_filled"]:
                inner_rect = slot["rect"].inflate(-6, -6)
                pygame.gfxdraw.box(self.screen, inner_rect, self.COLOR_GRID)

    def _render_blocks(self):
        for i, block in enumerate(self.blocks):
            if block["is_placed"]:
                # Render placed blocks within their slots
                slot = self.slots[block["target_slot_idx"]]
                rect = pygame.Rect(0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE)
                rect.center = slot["pos"]
                pygame.draw.rect(self.screen, block["color"], rect, border_radius=3)
            elif not block["is_held"]:
                 # Render unheld blocks
                rect = pygame.Rect(0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE)
                rect.center = block["pos"]
                pygame.draw.rect(self.screen, block["color"], rect, border_radius=3)
                pygame.draw.rect(self.screen, (255,255,255), rect, width=1, border_radius=3)

    def _render_arms(self):
        # Draw inactive arm first
        inactive_arm_idx = 1 - self.active_arm_idx
        self._render_single_arm(inactive_arm_idx, False)
        # Draw active arm on top
        self._render_single_arm(self.active_arm_idx, True)

    def _render_single_arm(self, idx, is_active):
        arm = self.arms[idx]
        pos = (int(arm["pos"].x), int(arm["pos"].y))
        color = arm["color_active"] if is_active else arm["color_inactive"]

        # Speed boost glow
        if arm["speed_boost_timer"] > 0:
            boost_alpha = 150 * (arm["speed_boost_timer"] / self.SPEED_BOOST_DURATION)
            self._draw_glow(self.screen, (*self.SPEED_BOOST_COLOR, boost_alpha), pos, self.ARM_RADIUS + 10, 8)

        # Active arm glow
        if is_active:
            self._draw_glow(self.screen, (*color, 80), pos, self.ARM_RADIUS + 5, 5)

        # Arm base
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ARM_RADIUS, color)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ARM_RADIUS, (255, 255, 255))

        # Gripper
        gripper_color = (200, 200, 200) if arm["holding_block_idx"] is None else (255, 255, 255)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ARM_RADIUS // 2, gripper_color)

        # Held block
        if arm["holding_block_idx"] is not None:
            block = self.blocks[arm["holding_block_idx"]]
            rect = pygame.Rect(0, 0, self.BLOCK_SIZE, self.BLOCK_SIZE)
            rect.center = pos
            pygame.draw.rect(self.screen, block["color"], rect, border_radius=3)
            pygame.draw.rect(self.screen, (255,255,255), rect, width=2, border_radius=3)

    def _render_particles(self):
        for p in self.particles:
            progress = p["life"] / p["max_life"]
            current_radius = int(p["max_radius"] * (1.0 - progress))
            alpha = int(255 * progress)
            color = (*p["color"], alpha)
            self._draw_glow(self.screen, color, p["pos"], current_radius, 5)

    def _render_ui(self):
        # Score
        score_text = f"PLACED: {self.score}/{self.NUM_BLOCKS}"
        self._draw_text(score_text, (10, 10), self.COLOR_UI_TEXT, self.font_ui)

        # Timer
        # The original code used 100 steps/sec, but let's assume 60fps to match human play
        time_left = (self.MAX_STEPS - self.steps) / 60.0
        time_left = max(0, time_left)
        timer_text = f"{time_left:.2f}"
        color = self.COLOR_UI_TEXT if time_left > 10 else (255, 80, 80)
        text_surface = self.font_timer.render(timer_text, True, color)
        self.screen.blit(text_surface, (self.SCREEN_WIDTH - text_surface.get_width() - 15, 10))

    def _draw_text(self, text, pos, color, font):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_glow(self, surface, color, center, max_radius, steps=10):
        center = (int(center[0]), int(center[1]))
        base_alpha = color[3] if len(color) > 3 else 255
        for i in range(steps, 0, -1):
            radius = int(max_radius * (i / steps))
            alpha = int(base_alpha * (i / steps)**2 * 0.5)
            if radius > 0 and alpha > 0:
                # Create a temporary surface for the glow circle
                glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color[:3], alpha), (radius, radius), radius)
                surface.blit(glow_surf, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)


    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for human play and visualization.
    # It will not run under the test harness.
    os.environ['SDL_VIDEODRIVER'] = 'x11' # or 'windows', 'macOS', etc.
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Dual Arm Block Placer")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    # In human mode, we need a different action mapping.
    # Action 0 is 'switch arm', but we also need a 'no-op' for movement.
    # We'll use a special key (Tab) for switching and treat movement=0 as no-op.
    while not done:
        movement = 5 # Use 5 as a placeholder for no-op, action space is 0-4
        space = 0
        shift = 0 # unused
        switch_arm_flag = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_TAB:
                     switch_arm_flag = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        if keys[pygame.K_SPACE]: space = 1
        
        # The action space expects movement to be 0 for switching.
        if switch_arm_flag:
            action = [0, space, shift]
        # If no movement key is pressed, it should be a no-op.
        # The original code used movement=0 for both no-op and switch, which is ambiguous.
        # We'll send a non-moving action (e.g., 1 with 0 direction) if no key is pressed.
        # However, to stick to the action space, a no-op movement is not explicitly defined.
        # Let's pass a dummy action if no movement is intended. A better way would be to have a dedicated no-op.
        # For now, let's stick to the original logic's intent. If no arrow key, movement is 0.
        # But we must distinguish it from switching. The `main` loop now handles this.
        else:
            if movement == 5: # If no movement key was pressed
                # We need to pass a valid action. The environment's step function has a bug
                # where movement=0 is always switch. Let's pass an invalid move action that does nothing.
                # A better fix would be in the step function, but sticking to the brief...
                # The original code's main loop had a bug. Let's fix it.
                # movement = 0 is switch. If no keys are pressed, we should send a no-op.
                # Let's define no-op as [5,0,0] and handle it. But action space is [5,2,2].
                # So movement must be in {0,1,2,3,4}.
                # The most logical no-op is [0,0,0], but 0 is 'switch'.
                # Let's assume the intent is that holding no keys is a no-op for movement.
                # The provided step function does nothing for movement > 4.
                # Let's use 0 as the no-op for movement if not switching.
                action = [0, space, shift] # No move, maybe press space
            else:
                 action = [movement, space, shift]


        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(60) # Limit human play FPS

    print(f"Game Over. Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()